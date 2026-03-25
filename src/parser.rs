use crate::definition::*;

/// Parse a unit definition file into a list of definitions.
pub fn parse_definitions(text: &str) -> Vec<Definition> {
    let mut defs = Vec::new();
    let mut in_defaults_block = false;
    let mut in_skip_block = false;
    let mut defaults_entries: Vec<(String, String)> = Vec::new();

    for raw_line in text.lines() {
        let line = raw_line.trim();

        // Skip empty lines
        if line.is_empty() {
            continue;
        }

        // Strip comments (but not inside strings)
        let line = strip_comment(line);
        let line = line.trim();
        if line.is_empty() {
            defs.push(Definition::Comment);
            continue;
        }

        // Handle @directives
        if line.starts_with('@') {
            if line == "@end" {
                if in_defaults_block {
                    defs.push(Definition::Defaults(defaults_entries.clone()));
                    defaults_entries.clear();
                    in_defaults_block = false;
                }
                in_skip_block = false;
                continue;
            }
            if line.starts_with("@defaults") {
                in_defaults_block = true;
                continue;
            }
            if line.starts_with("@system") {
                in_skip_block = true;
                continue;
            }
            if line.starts_with("@group") {
                // @group blocks contain real unit definitions, just skip the directive itself
                continue;
            }
            if let Some(rest) = line.strip_prefix("@import ") {
                defs.push(Definition::Import(rest.trim().to_string()));
                continue;
            }
            if let Some(rest) = line.strip_prefix("@alias ") {
                if let Some(def) = parse_alias(rest) {
                    defs.push(Definition::Alias(def));
                }
                continue;
            }
            continue;
        }

        // Inside defaults block
        if in_defaults_block {
            if let Some((key, val)) = line.split_once('=') {
                defaults_entries.push((key.trim().to_string(), val.trim().to_string()));
            }
            continue;
        }

        // Inside @group or @system block - skip these lines
        if in_skip_block {
            continue;
        }

        // Check if it's a prefix (name ends with -)
        if is_prefix_line(line) {
            if let Some(def) = parse_prefix(line) {
                defs.push(Definition::Prefix(def));
            }
            continue;
        }

        // Check if it's a dimension definition [name] = ...
        if line.starts_with('[') {
            if let Some(def) = parse_dimension(line) {
                defs.push(Definition::Dimension(def));
            }
            continue;
        }

        // Otherwise it's a unit definition
        if let Some(def) = parse_unit(line) {
            defs.push(Definition::Unit(def));
        }
    }

    defs
}

fn strip_comment(line: &str) -> &str {
    // Find the first # that's not inside a unit symbol
    if let Some(pos) = line.find('#') {
        line[..pos].trim_end()
    } else {
        line
    }
}

fn is_prefix_line(line: &str) -> bool {
    // A prefix line has the name ending with - before the =
    if let Some(eq_pos) = line.find('=') {
        let name_part = line[..eq_pos].trim();
        name_part.ends_with('-')
    } else {
        false
    }
}

fn parse_prefix(line: &str) -> Option<PrefixDef> {
    let (name_part, rest) = line.split_once('=')?;
    let name = name_part.trim().trim_end_matches('-').to_string();

    let parts: Vec<&str> = rest.splitn(2, '=').collect();
    let factor_str = parts[0].trim();

    let factor = eval_simple_expr(factor_str)?;

    let mut symbol = None;
    let mut aliases = Vec::new();

    if parts.len() > 1 {
        for part in parts[1].split('=') {
            let s = part.trim().trim_end_matches('-');
            if s == "_" {
                continue;
            }
            if s.is_empty() {
                continue;
            }
            if symbol.is_none() {
                symbol = Some(s.to_string());
            } else {
                aliases.push(s.to_string());
            }
        }
    }

    Some(PrefixDef {
        name,
        factor,
        symbol,
        aliases,
    })
}

fn parse_dimension(line: &str) -> Option<DimensionDef> {
    let (name_part, relation_part) = line.split_once('=')?;
    let name = name_part.trim().to_string();
    let relation = relation_part.trim().to_string();

    Some(DimensionDef {
        name,
        relation: if relation.is_empty() {
            None
        } else {
            Some(relation)
        },
    })
}

fn parse_alias(line: &str) -> Option<AliasDef> {
    let parts: Vec<&str> = line.split('=').collect();
    if parts.len() < 2 {
        return None;
    }
    let name = parts[0].trim().to_string();
    let aliases: Vec<String> = parts[1..]
        .iter()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty() && s != "_")
        .collect();

    Some(AliasDef { name, aliases })
}

fn parse_unit(line: &str) -> Option<UnitDef> {
    // Split on ; first to handle offset
    let (main_part, offset_part) = if let Some(pos) = line.find(';') {
        let (main, rest) = line.split_at(pos);
        (main, Some(&rest[1..]))
    } else {
        (line, None)
    };

    let offset = offset_part.and_then(|s| {
        // Format: "offset: <value> = ..."
        let s = s.trim();
        let s = s.strip_prefix("offset:")?;
        // Get just the number part, before any = signs
        let val_str = if let Some(eq_pos) = s.find('=') {
            s[..eq_pos].trim()
        } else {
            s.trim()
        };
        eval_simple_expr(val_str)
    });

    // Get aliases from after the offset's = signs too
    let mut extra_aliases_from_offset: Vec<String> = offset_part
        .map(|s| {
            let s = s.trim();
            // Everything after first = sign
            if let Some(pos) = s.find('=') {
                s[pos + 1..]
                    .split('=')
                    .map(|p| p.trim().to_string())
                    .filter(|p| !p.is_empty() && p != "_")
                    .collect()
            } else {
                Vec::new()
            }
        })
        .unwrap_or_default();

    // Now parse the main part: name = relation [= symbol] [= alias...]
    let eq_parts: Vec<&str> = main_part.split('=').collect();
    if eq_parts.is_empty() {
        return None;
    }

    let name = eq_parts[0].trim().to_string();

    // No = sign means it's a standalone name (unlikely but handle it)
    if eq_parts.len() < 2 {
        return Some(UnitDef {
            name,
            relation: String::new(),
            symbol: None,
            aliases: extra_aliases_from_offset,
            is_base: false,
            dimension: None,
            offset,
        });
    }

    let relation = eq_parts[1].trim().to_string();

    // Check if this defines a base dimension
    let is_base = relation.starts_with('[') && relation.ends_with(']');
    let dimension = if is_base {
        Some(relation.clone())
    } else {
        None
    };

    let mut symbol = None;
    let mut aliases = Vec::new();

    for part in &eq_parts[2..] {
        let s = part.trim();
        if s == "_" {
            // Skip placeholder symbol
            continue;
        }
        if s.is_empty() {
            continue;
        }
        if symbol.is_none() {
            symbol = Some(s.to_string());
        } else {
            aliases.push(s.to_string());
        }
    }

    // If no symbol from main part, use first alias from offset section
    if symbol.is_none() && !extra_aliases_from_offset.is_empty() {
        symbol = Some(extra_aliases_from_offset.remove(0));
    }
    aliases.extend(extra_aliases_from_offset);

    Some(UnitDef {
        name,
        relation,
        symbol,
        aliases,
        is_base,
        dimension,
        offset,
    })
}

/// Evaluate simple numeric expressions like "1e-10", "2**10", "1e3", "64.79891",
/// "π / 180", "5 / 9", "233.15 + 200 / 9"
pub fn eval_simple_expr(s: &str) -> Option<f64> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }

    // Try simple float first
    if let Ok(v) = s.parse::<f64>() {
        return Some(v);
    }

    // Handle π
    let s = s.replace('π', "3.1415926535897932384626433832795028841971693993751");

    // Tokenize and evaluate with basic precedence
    eval_expression(&s)
}

fn eval_expression(s: &str) -> Option<f64> {
    let tokens = tokenize(s)?;
    eval_tokens(&tokens, 0).map(|(v, _)| v)
}

#[derive(Debug, Clone)]
enum Token {
    Num(f64),
    Op(char),
    LParen,
    RParen,
}

fn tokenize(s: &str) -> Option<Vec<Token>> {
    let mut tokens = Vec::new();
    let chars: Vec<char> = s.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        match chars[i] {
            ' ' | '\t' => {
                i += 1;
            }
            '(' => {
                tokens.push(Token::LParen);
                i += 1;
            }
            ')' => {
                tokens.push(Token::RParen);
                i += 1;
            }
            '+' | '/' => {
                tokens.push(Token::Op(chars[i]));
                i += 1;
            }
            '-' => {
                // Could be unary minus or subtraction
                let is_unary = tokens.is_empty()
                    || matches!(tokens.last(), Some(Token::Op(_) | Token::LParen));
                if is_unary {
                    // Parse as part of a number
                    let start = i;
                    i += 1;
                    while i < chars.len()
                        && (chars[i].is_ascii_digit()
                            || chars[i] == '.'
                            || chars[i] == 'e'
                            || chars[i] == 'E')
                    {
                        i += 1;
                    }
                    let num_str: String = chars[start..i].iter().collect();
                    tokens.push(Token::Num(num_str.parse().ok()?));
                } else {
                    tokens.push(Token::Op('-'));
                    i += 1;
                }
            }
            '*' => {
                if i + 1 < chars.len() && chars[i + 1] == '*' {
                    tokens.push(Token::Op('^'));
                    i += 2;
                } else {
                    tokens.push(Token::Op('*'));
                    i += 1;
                }
            }
            c if c.is_ascii_digit() || c == '.' => {
                let start = i;
                while i < chars.len()
                    && (chars[i].is_ascii_digit()
                        || chars[i] == '.'
                        || chars[i] == 'e'
                        || chars[i] == 'E'
                        || ((chars[i] == '+' || chars[i] == '-')
                            && i > start
                            && (chars[i - 1] == 'e' || chars[i - 1] == 'E')))
                {
                    i += 1;
                }
                let num_str: String = chars[start..i].iter().collect();
                tokens.push(Token::Num(num_str.parse().ok()?));
            }
            _ => {
                // Unknown char, skip
                i += 1;
            }
        }
    }

    Some(tokens)
}

/// Simple recursive descent expression evaluator with precedence:
/// Level 0: + -
/// Level 1: * /
/// Level 2: ^
/// Level 3: atoms (numbers, parenthesized expressions)
fn eval_tokens(tokens: &[Token], pos: usize) -> Option<(f64, usize)> {
    eval_additive(tokens, pos)
}

fn eval_additive(tokens: &[Token], pos: usize) -> Option<(f64, usize)> {
    let (mut left, mut pos) = eval_multiplicative(tokens, pos)?;
    while pos < tokens.len() {
        match &tokens[pos] {
            Token::Op('+') => {
                let (right, new_pos) = eval_multiplicative(tokens, pos + 1)?;
                left += right;
                pos = new_pos;
            }
            Token::Op('-') => {
                let (right, new_pos) = eval_multiplicative(tokens, pos + 1)?;
                left -= right;
                pos = new_pos;
            }
            _ => break,
        }
    }
    Some((left, pos))
}

fn eval_multiplicative(tokens: &[Token], pos: usize) -> Option<(f64, usize)> {
    let (mut left, mut pos) = eval_power(tokens, pos)?;
    while pos < tokens.len() {
        match &tokens[pos] {
            Token::Op('*') => {
                let (right, new_pos) = eval_power(tokens, pos + 1)?;
                left *= right;
                pos = new_pos;
            }
            Token::Op('/') => {
                let (right, new_pos) = eval_power(tokens, pos + 1)?;
                left /= right;
                pos = new_pos;
            }
            _ => break,
        }
    }
    Some((left, pos))
}

fn eval_power(tokens: &[Token], pos: usize) -> Option<(f64, usize)> {
    let (base, mut pos) = eval_atom(tokens, pos)?;
    if pos < tokens.len() {
        if let Token::Op('^') = &tokens[pos] {
            let (exp, new_pos) = eval_power(tokens, pos + 1)?;
            pos = new_pos;
            return Some((base.powf(exp), pos));
        }
    }
    Some((base, pos))
}

fn eval_atom(tokens: &[Token], pos: usize) -> Option<(f64, usize)> {
    if pos >= tokens.len() {
        return None;
    }
    match &tokens[pos] {
        Token::Num(v) => Some((*v, pos + 1)),
        Token::LParen => {
            let (val, new_pos) = eval_additive(tokens, pos + 1)?;
            // expect RParen
            if new_pos < tokens.len() {
                if let Token::RParen = &tokens[new_pos] {
                    return Some((val, new_pos + 1));
                }
            }
            Some((val, new_pos))
        }
        Token::Op('-') => {
            let (val, new_pos) = eval_atom(tokens, pos + 1)?;
            Some((-val, new_pos))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_prefix() {
        let defs = parse_definitions("kilo- = 1e3 = k-");
        assert_eq!(defs.len(), 1);
        if let Definition::Prefix(p) = &defs[0] {
            assert_eq!(p.name, "kilo");
            assert_eq!(p.factor, 1e3);
            assert_eq!(p.symbol.as_deref(), Some("k"));
        } else {
            panic!("Expected prefix");
        }
    }

    #[test]
    fn test_parse_base_unit() {
        let defs = parse_definitions("meter = [length] = m = metre");
        assert_eq!(defs.len(), 1);
        if let Definition::Unit(u) = &defs[0] {
            assert_eq!(u.name, "meter");
            assert!(u.is_base);
            assert_eq!(u.symbol.as_deref(), Some("m"));
            assert!(u.aliases.contains(&"metre".to_string()));
        } else {
            panic!("Expected unit");
        }
    }

    #[test]
    fn test_parse_derived_unit() {
        let defs = parse_definitions("minute = 60 * second = min");
        assert_eq!(defs.len(), 1);
        if let Definition::Unit(u) = &defs[0] {
            assert_eq!(u.name, "minute");
            assert_eq!(u.relation, "60 * second");
            assert_eq!(u.symbol.as_deref(), Some("min"));
        } else {
            panic!("Expected unit");
        }
    }

    #[test]
    fn test_eval_simple() {
        assert_eq!(eval_simple_expr("1e3"), Some(1e3));
        assert_eq!(eval_simple_expr("2**10"), Some(1024.0));
        assert!((eval_simple_expr("5 / 9").unwrap() - 5.0 / 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_parse_offset_unit() {
        let defs =
            parse_definitions("degree_Celsius = kelvin; offset: 273.15 = °C = celsius = degC");
        if let Definition::Unit(u) = &defs[0] {
            assert_eq!(u.name, "degree_Celsius");
            assert_eq!(u.relation, "kelvin");
            assert!((u.offset.unwrap() - 273.15).abs() < 1e-10);
            assert_eq!(u.symbol.as_deref(), Some("°C"));
        } else {
            panic!("Expected unit");
        }
    }
}
