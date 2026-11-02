// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! SQL `LIKE` / `ILIKE` kernels with the semantics of the arrow 13 kernels
//! (`like_utf8`, `like_utf8_scalar`, `ilike_utf8`, ...) DataFusion was built on.
//!
//! arrow 60's `like` / `ilike` kernels differ from them: a backslash escapes any
//! character (arrow 13 only treated `\%` and `\_` as escapes and kept every other
//! backslash literally), `_` and `%` match newlines, and patterns without wildcards are
//! not compared with plain string equality. These kernels keep the arrow 13 results:
//!
//! * a scalar pattern without `%` and `_` is compared with `==` (for `ILIKE` too, which
//!   makes such an `ILIKE` case-sensitive, like arrow 13);
//! * a scalar pattern `prefix%` / `%suffix` without other wildcards uses `starts_with` /
//!   `ends_with` (`ILIKE` compares the `to_uppercase` forms);
//! * every other pattern is translated to the regular expression `^...$` by
//!   [`like_pattern_to_regex`] and matched with arrow's regex kernels (`(?i)` for `ILIKE`).

use arrow::array::{Array, BooleanArray, GenericStringArray, OffsetSizeTrait};
use arrow::buffer::BooleanBuffer;
use arrow::compute::kernels::boolean::not;
use arrow::compute::kernels::regexp::{regexp_is_match, regexp_is_match_scalar};
use arrow::error::ArrowError;

/// Returns true if `c` is a `LIKE` wildcard
fn is_like_pattern(c: char) -> bool {
    c == '%' || c == '_'
}

/// Returns true if `c` has a special meaning in a regular expression
/// (`regex_syntax::is_meta_character`)
fn is_regex_meta_character(c: char) -> bool {
    matches!(
        c,
        '\\' | '.'
            | '+'
            | '*'
            | '?'
            | '('
            | ')'
            | '|'
            | '['
            | ']'
            | '{'
            | '}'
            | '^'
            | '$'
            | '#'
            | '&'
            | '-'
            | '~'
    )
}

/// Translates a `LIKE` pattern to an anchored regular expression, like arrow 13's
/// `replace_like_wildcards`:
///
/// 1. `%` becomes `.*` and `_` becomes `.` (neither matches a newline);
/// 2. regex meta characters are escaped;
/// 3. `\%` and `\_` match a literal `%` / `_`; any other backslash is a literal backslash.
pub(crate) fn like_pattern_to_regex(pattern: &str) -> String {
    let mut result = String::with_capacity(pattern.len() + 2);
    result.push('^');
    let mut chars = pattern.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.peek() {
                Some(next) if is_like_pattern(*next) => {
                    result.push(*next);
                    chars.next();
                }
                _ => result.push_str("\\\\"),
            }
        } else if is_regex_meta_character(c) {
            result.push('\\');
            result.push(c);
        } else if c == '%' {
            result.push_str(".*");
        } else if c == '_' {
            result.push('.');
        } else {
            result.push(c);
        }
    }
    result.push('$');
    result
}

/// Applies `matches` to every value of `array` (null slots included, their result is
/// masked by the nulls of `array`)
fn map_values<O: OffsetSizeTrait>(
    array: &GenericStringArray<O>,
    matches: impl Fn(&str) -> bool,
) -> BooleanArray {
    let values = BooleanBuffer::collect_bool(array.len(), |i| matches(array.value(i)));
    BooleanArray::new(values, array.nulls().cloned())
}

/// `array [NOT] [I]LIKE pattern` for a scalar `pattern`
pub(crate) fn like_scalar<O: OffsetSizeTrait>(
    array: &GenericStringArray<O>,
    pattern: &str,
    case_insensitive: bool,
    negated: bool,
) -> Result<BooleanArray, ArrowError> {
    let result = if !pattern.contains(is_like_pattern) {
        // fast path, arrow 13 used plain equality, even for `ILIKE`
        map_values(array, |v| v == pattern)
    } else if pattern.ends_with('%')
        && !pattern.ends_with("\\%")
        && !pattern[..pattern.len() - 1].contains(is_like_pattern)
    {
        // fast path, starts_with
        let prefix = &pattern[..pattern.len() - 1];
        if case_insensitive {
            let prefix = prefix.to_uppercase();
            map_values(array, |v| v.to_uppercase().starts_with(&prefix))
        } else {
            map_values(array, |v| v.starts_with(prefix))
        }
    } else if pattern.starts_with('%') && !pattern[1..].contains(is_like_pattern) {
        // fast path, ends_with
        let suffix = &pattern[1..];
        if case_insensitive {
            let suffix = suffix.to_uppercase();
            map_values(array, |v| v.to_uppercase().ends_with(&suffix))
        } else {
            map_values(array, |v| v.ends_with(suffix))
        }
    } else {
        regexp_is_match_scalar(
            array,
            &like_pattern_to_regex(pattern),
            if case_insensitive { Some("i") } else { None },
        )?
    };
    if negated {
        not(&result)
    } else {
        Ok(result)
    }
}

/// `left [NOT] [I]LIKE right` for two arrays of the same length. Like arrow 13, every
/// pattern is matched as a regular expression (no fast paths) and a row is null when
/// either side is null.
pub(crate) fn like_arrays<O: OffsetSizeTrait>(
    left: &GenericStringArray<O>,
    right: &GenericStringArray<O>,
    case_insensitive: bool,
    negated: bool,
) -> Result<BooleanArray, ArrowError> {
    let patterns = right
        .iter()
        .map(|pattern| pattern.map(like_pattern_to_regex))
        .collect::<GenericStringArray<O>>();
    let result = if case_insensitive {
        let flags = std::iter::repeat(Some("i"))
            .take(left.len())
            .collect::<GenericStringArray<O>>();
        regexp_is_match(left, &patterns, Some(&flags))?
    } else {
        regexp_is_match(left, &patterns, None::<&GenericStringArray<O>>)?
    };
    if negated {
        not(&result)
    } else {
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::StringArray;

    fn scalar(values: Vec<Option<&str>>, pattern: &str, ci: bool) -> BooleanArray {
        like_scalar(&StringArray::from(values), pattern, ci, false).unwrap()
    }

    fn arrays(
        values: Vec<Option<&str>>,
        patterns: Vec<Option<&str>>,
        ci: bool,
    ) -> BooleanArray {
        like_arrays(
            &StringArray::from(values),
            &StringArray::from(patterns),
            ci,
            false,
        )
        .unwrap()
    }

    #[test]
    fn like_pattern_translation() {
        assert_eq!(like_pattern_to_regex("a%b_c"), "^a.*b.c$");
        assert_eq!(like_pattern_to_regex("a.b"), "^a\\.b$");
        assert_eq!(like_pattern_to_regex("a\\%b\\_"), "^a%b_$");
        // a backslash not followed by a wildcard is a literal backslash
        assert_eq!(like_pattern_to_regex("a\\b"), "^a\\\\b$");
        // `\\%` is a literal backslash followed by an escaped (literal) `%`
        assert_eq!(like_pattern_to_regex("C:\\\\%"), "^C:\\\\%$");
        assert_eq!(like_pattern_to_regex("a\\"), "^a\\\\$");
    }

    #[test]
    fn like_scalar_arrow13_semantics() {
        // no wildcards: plain equality, backslashes are literal
        assert_eq!(
            scalar(vec![Some("a\\b"), Some("ab"), None], "a\\b", false),
            BooleanArray::from(vec![Some(true), Some(false), None])
        );
        // `\\` followed by `%` is a literal backslash and an escaped `%`
        assert_eq!(
            scalar(vec![Some("C:\\temp"), Some("C:\\%")], "C:\\\\%", false),
            BooleanArray::from(vec![false, true])
        );
        // `_` and `%` do not match a newline
        assert_eq!(
            scalar(vec![Some("a\nb"), Some("axb")], "a_b", false),
            BooleanArray::from(vec![false, true])
        );
        assert_eq!(
            scalar(vec![Some("a\nb"), Some("axxb")], "a%b", false),
            BooleanArray::from(vec![false, true])
        );
        // prefix / suffix fast paths
        assert_eq!(
            scalar(vec![Some("arrow"), Some("parrot"), None], "arr%", false),
            BooleanArray::from(vec![Some(true), Some(false), None])
        );
        assert_eq!(
            scalar(vec![Some("arrow"), Some("parrot")], "%rot", false),
            BooleanArray::from(vec![false, true])
        );
        assert_eq!(
            scalar(vec![Some("100%"), Some("1000")], "100\\%", false),
            BooleanArray::from(vec![true, false])
        );
    }

    #[test]
    fn ilike_scalar_arrow13_semantics() {
        // prefix fast path compares the upper case forms
        assert_eq!(
            scalar(vec![Some("STRASSE"), Some("strasse x")], "straße%", true),
            BooleanArray::from(vec![true, true])
        );
        assert_eq!(
            scalar(vec![Some("ARROW"), Some("parrot")], "%ROW", true),
            BooleanArray::from(vec![true, false])
        );
        assert_eq!(
            scalar(vec![Some("ArRoW"), Some("parrot")], "a_r%", true),
            BooleanArray::from(vec![true, false])
        );
        // arrow 13 compared patterns without wildcards with `==`, even for ILIKE
        assert_eq!(
            scalar(vec![Some("abc"), Some("ABC")], "abc", true),
            BooleanArray::from(vec![true, false])
        );
        // negation keeps nulls
        assert_eq!(
            like_scalar(
                &StringArray::from(vec![Some("ARROW"), Some("x"), None]),
                "arr%",
                true,
                true
            )
            .unwrap(),
            BooleanArray::from(vec![Some(false), Some(true), None])
        );
    }

    #[test]
    fn like_arrays_arrow13_semantics() {
        assert_eq!(
            arrays(
                vec![
                    Some("a\\b"),
                    Some("a\nb"),
                    Some("C:\\temp"),
                    None,
                    Some("x")
                ],
                vec![Some("a\\b"), Some("a_b"), Some("C:\\\\%"), Some("%"), None],
                false
            ),
            BooleanArray::from(vec![Some(true), Some(false), Some(false), None, None])
        );
        assert_eq!(
            arrays(
                vec![Some("ABC"), Some("Arrow")],
                vec![Some("abc"), Some("a%W")],
                true
            ),
            BooleanArray::from(vec![true, true])
        );
        assert_eq!(
            like_arrays(
                &StringArray::from(vec![Some("abc"), Some("abd"), None]),
                &StringArray::from(vec![Some("ab_"), Some("abc"), Some("x")]),
                false,
                true
            )
            .unwrap(),
            BooleanArray::from(vec![Some(false), Some(true), None])
        );
    }
}
