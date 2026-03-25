use std::collections::HashMap;

use crate::definition::*;
use crate::errors::*;
use crate::parser;
use crate::units_container::UnitsContainer;

const UNIT_CACHE_MAX: usize = 512;

/// Internal representation of a unit in the registry.
#[derive(Debug, Clone)]
pub struct UnitEntry {
    pub name: String,
    /// Conversion factor to the reference unit (in root units)
    pub factor: f64,
    /// The dimensions of this unit expressed as root units
    pub root_units: UnitsContainer,
    /// The dimensionality expressed as base dimensions like [length], [time]
    pub dimensionality: UnitsContainer,
    /// Offset for non-multiplicative units (e.g. degC offset: 273.15)
    pub offset: f64,
    pub symbol: Option<String>,
    pub aliases: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct PrefixEntry {
    pub factor: f64,
    pub symbol: Option<String>,
    pub aliases: Vec<String>,
}

/// The UnitRegistry holds all unit definitions, prefixes, and conversion logic.
pub struct UnitRegistry {
    /// Map from canonical name -> UnitEntry
    units: HashMap<String, UnitEntry>,
    /// Map from prefix name -> PrefixEntry
    prefixes: HashMap<String, PrefixEntry>,
    /// Map from any name (symbol, alias, plural) -> canonical name
    name_map: HashMap<String, String>,
    /// Map from prefix symbol/alias -> canonical prefix name
    prefix_map: HashMap<String, String>,
    /// Raw definitions (needed for multi-pass resolution)
    raw_unit_defs: Vec<UnitDef>,
    /// Map from dimension name like "[length]" -> base dimension or derived
    dimensions: HashMap<String, Option<String>>,
    /// Whether the registry is case sensitive
    pub case_sensitive: bool,
    /// Cache for parsed unit expressions
    unit_expr_cache: HashMap<String, UnitsContainer>,
    /// Cache for unit info: name -> (factor, root_units, dimensionality)
    unit_info_cache: HashMap<String, (f64, UnitsContainer, UnitsContainer)>,
    /// Pre-sorted prefix strings (longest first) for fast prefix stripping
    sorted_prefixes: Vec<(String, f64)>,
}

impl UnitRegistry {
    pub fn new() -> PintResult<Self> {
        let mut reg = Self {
            units: HashMap::new(),
            prefixes: HashMap::new(),
            name_map: HashMap::new(),
            prefix_map: HashMap::new(),
            raw_unit_defs: Vec::new(),
            dimensions: HashMap::new(),
            case_sensitive: true,
            unit_expr_cache: HashMap::new(),
            unit_info_cache: HashMap::new(),
            sorted_prefixes: Vec::new(),
        };

        // Load embedded default definitions
        let default_txt = include_str!("default_en.txt");
        let constants_txt = include_str!("constants_en.txt");

        // Parse definitions
        let mut all_defs = parser::parse_definitions(default_txt);

        // Resolve @import directives
        let mut resolved = Vec::new();
        for def in all_defs.drain(..) {
            match def {
                Definition::Import(ref filename) if filename == "constants_en.txt" => {
                    let const_defs = parser::parse_definitions(constants_txt);
                    resolved.extend(const_defs);
                }
                _ => resolved.push(def),
            }
        }

        // First pass: register prefixes
        for def in &resolved {
            if let Definition::Prefix(p) = def {
                reg.add_prefix(p);
            }
        }

        // Build sorted prefix list for fast prefix stripping
        reg.rebuild_sorted_prefixes();

        // Collect unit definitions for multi-pass resolution
        let mut unit_defs = Vec::new();
        let mut alias_defs = Vec::new();

        for def in &resolved {
            match def {
                Definition::Unit(u) => {
                    unit_defs.push(u.clone());
                }
                Definition::Dimension(d) => {
                    reg.dimensions.insert(d.name.clone(), d.relation.clone());
                }
                Definition::Alias(a) => {
                    alias_defs.push(a.clone());
                }
                _ => {}
            }
        }

        reg.raw_unit_defs = unit_defs.clone();

        // Multi-pass: resolve units (some depend on others not yet defined)
        let mut remaining = unit_defs;
        let max_passes = 20;
        for _ in 0..max_passes {
            if remaining.is_empty() {
                break;
            }
            let prev_len = remaining.len();
            let mut unresolved = Vec::new();
            for def in remaining.drain(..) {
                match reg.try_add_unit(&def) {
                    Ok(()) => {}
                    Err(_) => {
                        unresolved.push(def);
                    }
                }
            }
            if unresolved.len() == prev_len {
                break;
            }
            remaining = unresolved;
        }

        // Process aliases
        for alias_def in &alias_defs {
            if let Some(canonical) = reg.name_map.get(&alias_def.name).cloned() {
                for alias in &alias_def.aliases {
                    reg.name_map.insert(alias.clone(), canonical.clone());
                    reg.name_map
                        .insert(format!("{}s", alias), canonical.clone());
                }
            }
        }

        Ok(reg)
    }

    fn add_prefix(&mut self, p: &PrefixDef) {
        let entry = PrefixEntry {
            factor: p.factor,
            symbol: p.symbol.clone(),
            aliases: p.aliases.clone(),
        };

        self.prefix_map.insert(p.name.clone(), p.name.clone());
        if let Some(sym) = &p.symbol {
            self.prefix_map.insert(sym.clone(), p.name.clone());
        }
        for alias in &p.aliases {
            self.prefix_map.insert(alias.clone(), p.name.clone());
        }

        self.prefixes.insert(p.name.clone(), entry);
    }

    fn try_add_unit(&mut self, def: &UnitDef) -> PintResult<()> {
        if def.is_base {
            // Base unit: defines a new dimension
            let dim_name = def.dimension.as_ref().unwrap().clone();
            let entry = UnitEntry {
                name: def.name.clone(),
                factor: 1.0,
                root_units: UnitsContainer::from_single(def.name.clone(), 1.0),
                dimensionality: UnitsContainer::from_single(dim_name.clone(), 1.0),
                offset: def.offset.unwrap_or(0.0),
                symbol: def.symbol.clone(),
                aliases: def.aliases.clone(),
            };
            self.register_unit(entry);
            self.dimensions.insert(dim_name, None);
            return Ok(());
        }

        // Derived unit: resolve the relation
        let (factor, root_units, dimensionality) = self.resolve_relation(&def.relation)?;

        let entry = UnitEntry {
            name: def.name.clone(),
            factor,
            root_units,
            dimensionality,
            offset: def.offset.unwrap_or(0.0),
            symbol: def.symbol.clone(),
            aliases: def.aliases.clone(),
        };

        self.register_unit(entry);
        Ok(())
    }

    fn register_unit(&mut self, entry: UnitEntry) {
        let canonical = entry.name.clone();

        // Register all name variants -> canonical name
        self.name_map.insert(canonical.clone(), canonical.clone());
        // Plural
        self.name_map
            .insert(format!("{}s", canonical), canonical.clone());

        if let Some(ref sym) = entry.symbol {
            self.name_map.insert(sym.clone(), canonical.clone());
        }
        for alias in &entry.aliases {
            self.name_map.insert(alias.clone(), canonical.clone());
            // Plural of aliases
            self.name_map
                .insert(format!("{}s", alias), canonical.clone());
        }

        self.units.insert(canonical, entry);
    }

    /// Resolve a relation string like "1e-10 * meter", "60 * second",
    /// "kilogram * meter / second ** 2", "[mass] * [length] / [time] ** 2",
    /// or just "micrometer" into (factor, root_units, dimensionality).
    fn resolve_relation(
        &self,
        relation: &str,
    ) -> PintResult<(f64, UnitsContainer, UnitsContainer)> {
        let relation = relation.trim();

        // Special case: empty relation = dimensionless
        if relation.is_empty() || relation == "[]" {
            return Ok((1.0, UnitsContainer::new(), UnitsContainer::new()));
        }

        // Tokenize the relation into numbers, unit references, and operators
        let tokens = tokenize_relation(relation);
        self.eval_relation_tokens(&tokens)
    }

    fn eval_relation_tokens(
        &self,
        tokens: &[RelToken],
    ) -> PintResult<(f64, UnitsContainer, UnitsContainer)> {
        let mut factor = 1.0_f64;
        let mut root_units = UnitsContainer::new();
        let mut dimensionality = UnitsContainer::new();
        let mut pending_op = RelOp::Mul;

        let mut i = 0;
        while i < tokens.len() {
            match &tokens[i] {
                RelToken::Number(n) => {
                    let mut val = *n;
                    // Check for ** (power) right after number
                    if i + 2 < tokens.len() {
                        if let RelToken::Op(RelOp::Pow) = &tokens[i + 1] {
                            if let RelToken::Number(exp) = &tokens[i + 2] {
                                val = val.powf(*exp);
                                i += 2;
                            }
                        }
                    }
                    match pending_op {
                        RelOp::Mul => factor *= val,
                        RelOp::Div => factor /= val,
                        RelOp::Pow => factor = factor.powf(val),
                    }
                }
                RelToken::UnitRef(name) => {
                    // Check for ** (power) right after unit
                    let mut power = 1.0;
                    if i + 2 < tokens.len() {
                        if let RelToken::Op(RelOp::Pow) = &tokens[i + 1] {
                            if let RelToken::Number(exp) = &tokens[i + 2] {
                                power = *exp;
                                i += 2;
                            }
                        }
                    }

                    // Check if name is a dimension reference like [length]
                    if name.starts_with('[') && name.ends_with(']') {
                        let dim_uc = UnitsContainer::from_single(name.clone(), power);
                        match pending_op {
                            RelOp::Mul => {
                                dimensionality = &dimensionality * &dim_uc;
                            }
                            RelOp::Div => {
                                dimensionality = &dimensionality / &dim_uc;
                            }
                            _ => {}
                        }
                    } else {
                        // Resolve the unit reference
                        let (uf, uru, udim) = self.resolve_unit_ref(name)?;
                        match pending_op {
                            RelOp::Mul => {
                                factor *= uf.powf(power);
                                root_units = &root_units * &uru.pow(power);
                                dimensionality = &dimensionality * &udim.pow(power);
                            }
                            RelOp::Div => {
                                factor /= uf.powf(power);
                                root_units = &root_units / &uru.pow(power);
                                dimensionality = &dimensionality / &udim.pow(power);
                            }
                            RelOp::Pow => {
                                // Unusual but handle it
                                factor = factor.powf(uf.powf(power));
                            }
                        }
                    }
                }
                RelToken::Op(op) => {
                    pending_op = *op;
                }
                RelToken::LParen => {
                    // Find matching RParen and recurse
                    let mut depth = 1;
                    let start = i + 1;
                    let mut j = start;
                    while j < tokens.len() && depth > 0 {
                        match &tokens[j] {
                            RelToken::LParen => depth += 1,
                            RelToken::RParen => depth -= 1,
                            _ => {}
                        }
                        if depth > 0 {
                            j += 1;
                        }
                    }
                    let sub_tokens = &tokens[start..j];
                    let (sf, sru, sdim) = self.eval_relation_tokens(sub_tokens)?;

                    // Check for ** after parens
                    let mut power = 1.0;
                    if j + 2 < tokens.len() {
                        if let RelToken::Op(RelOp::Pow) = &tokens[j + 1] {
                            if let RelToken::Number(exp) = &tokens[j + 2] {
                                power = *exp;
                                j += 2;
                            }
                        }
                    }

                    match pending_op {
                        RelOp::Mul => {
                            factor *= sf.powf(power);
                            root_units = &root_units * &sru.pow(power);
                            dimensionality = &dimensionality * &sdim.pow(power);
                        }
                        RelOp::Div => {
                            factor /= sf.powf(power);
                            root_units = &root_units / &sru.pow(power);
                            dimensionality = &dimensionality / &sdim.pow(power);
                        }
                        _ => {}
                    }
                    i = j;
                }
                RelToken::RParen => {
                    // Should not happen at top level
                }
            }
            i += 1;
        }

        Ok((factor, root_units, dimensionality))
    }

    /// Resolve a single unit name (possibly prefixed) to its factor, root_units, dimensionality
    fn resolve_unit_ref(&self, name: &str) -> PintResult<(f64, UnitsContainer, UnitsContainer)> {
        // First try direct lookup
        if let Some(canonical) = self.name_map.get(name) {
            if let Some(entry) = self.units.get(canonical) {
                return Ok((
                    entry.factor,
                    entry.root_units.clone(),
                    entry.dimensionality.clone(),
                ));
            }
        }

        // Try prefixed unit
        if let Some((prefix_factor, base_canonical)) = self.try_strip_prefix(name) {
            if let Some(entry) = self.units.get(&base_canonical) {
                return Ok((
                    prefix_factor * entry.factor,
                    entry.root_units.clone(),
                    entry.dimensionality.clone(),
                ));
            }
        }

        Err(Box::new(PintError::UndefinedUnitError {
            unit_name: name.to_string(),
        }))
    }

    fn rebuild_sorted_prefixes(&mut self) {
        let mut sp: Vec<(String, f64)> = Vec::new();
        for (name, entry) in &self.prefixes {
            if !name.is_empty() {
                sp.push((name.clone(), entry.factor));
            }
            if let Some(sym) = &entry.symbol {
                if !sym.is_empty() {
                    sp.push((sym.clone(), entry.factor));
                }
            }
            for alias in &entry.aliases {
                if !alias.is_empty() {
                    sp.push((alias.clone(), entry.factor));
                }
            }
        }
        sp.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
        self.sorted_prefixes = sp;
    }

    /// Try to strip a prefix from a unit name.
    /// Returns (prefix_factor, canonical_base_unit_name) or None.
    #[inline]
    fn try_strip_prefix(&self, name: &str) -> Option<(f64, String)> {
        for (prefix_str, factor) in &self.sorted_prefixes {
            if name.len() > prefix_str.len() && name.starts_with(prefix_str.as_str()) {
                let remainder = &name[prefix_str.len()..];
                if let Some(canonical) = self.name_map.get(remainder) {
                    if self.units.contains_key(canonical) {
                        return Some((*factor, canonical.clone()));
                    }
                }
            }
        }
        None
    }

    // --- Public API ---

    /// Get the canonical name for a unit string
    pub fn get_canonical_name(&self, name: &str) -> PintResult<String> {
        let lookup = if self.case_sensitive {
            name.to_string()
        } else {
            name.to_lowercase()
        };

        if let Some(canonical) = self.name_map.get(&lookup) {
            return Ok(canonical.clone());
        }

        // Try prefixed
        if let Some((_, canonical)) = self.try_strip_prefix(&lookup) {
            return Ok(format!("{}(prefixed)", canonical));
        }

        Err(Box::new(PintError::UndefinedUnitError {
            unit_name: name.to_string(),
        }))
    }

    /// Parse a unit expression string like "kg*m/s^2" into a UnitsContainer (canonical names).
    pub fn parse_unit_expr(&mut self, expr: &str) -> PintResult<UnitsContainer> {
        let expr = expr.trim();
        if expr.is_empty() || expr == "dimensionless" {
            return Ok(UnitsContainer::new());
        }

        if let Some(cached) = self.unit_expr_cache.get(expr) {
            return Ok(cached.clone());
        }

        let tokens = tokenize_relation(expr);
        let result = self.eval_unit_expr_tokens(&tokens)?;

        if self.unit_expr_cache.len() < UNIT_CACHE_MAX {
            self.unit_expr_cache.insert(expr.to_string(), result.clone());
        }

        Ok(result)
    }

    /// Evaluate unit expression tokens into a UnitsContainer
    fn eval_unit_expr_tokens(&self, tokens: &[RelToken]) -> PintResult<UnitsContainer> {
        let mut result = UnitsContainer::new();
        let mut pending_op = RelOp::Mul;
        let mut pending_factor = 1.0_f64;

        let mut i = 0;
        while i < tokens.len() {
            match &tokens[i] {
                RelToken::Number(n) => {
                    // Numbers in unit expressions are usually exponents or scale factors
                    // Check if next token is a power op
                    pending_factor *= n;
                }
                RelToken::UnitRef(name) => {
                    let canonical = self.resolve_to_canonical(name)?;
                    let mut power = 1.0;
                    if i + 2 < tokens.len() {
                        if let RelToken::Op(RelOp::Pow) = &tokens[i + 1] {
                            if let RelToken::Number(exp) = &tokens[i + 2] {
                                power = *exp;
                                i += 2;
                            }
                        }
                    }
                    match pending_op {
                        RelOp::Mul => {
                            result = result.add(&canonical, power);
                        }
                        RelOp::Div => {
                            result = result.add(&canonical, -power);
                        }
                        _ => {}
                    }
                }
                RelToken::Op(op) => {
                    pending_op = *op;
                }
                RelToken::LParen => {
                    let mut depth = 1;
                    let start = i + 1;
                    let mut j = start;
                    while j < tokens.len() && depth > 0 {
                        match &tokens[j] {
                            RelToken::LParen => depth += 1,
                            RelToken::RParen => depth -= 1,
                            _ => {}
                        }
                        if depth > 0 {
                            j += 1;
                        }
                    }
                    let sub = self.eval_unit_expr_tokens(&tokens[start..j])?;
                    let mut power = 1.0;
                    if j + 2 < tokens.len() {
                        if let RelToken::Op(RelOp::Pow) = &tokens[j + 1] {
                            if let RelToken::Number(exp) = &tokens[j + 2] {
                                power = *exp;
                                j += 2;
                            }
                        }
                    }
                    let sub = sub.pow(power);
                    match pending_op {
                        RelOp::Mul => result = &result * &sub,
                        RelOp::Div => result = &result / &sub,
                        _ => {}
                    }
                    i = j;
                }
                RelToken::RParen => {}
            }
            i += 1;
        }

        let _ = pending_factor; // Scale factors in unit expressions are handled during conversion
        Ok(result)
    }

    /// Resolve a unit name (possibly prefixed) to its canonical name
    fn resolve_to_canonical(&self, name: &str) -> PintResult<String> {
        // Direct lookup
        if let Some(canonical) = self.name_map.get(name) {
            return Ok(canonical.clone());
        }

        // Try prefixed - store as a prefixed unit
        if let Some((_prefix_factor, _base_canonical)) = self.try_strip_prefix(name) {
            return Ok(name.to_string());
        }

        Err(Box::new(PintError::UndefinedUnitError {
            unit_name: name.to_string(),
        }))
    }

    /// Get the dimensionality of a units container
    pub fn get_dimensionality(&mut self, units: &UnitsContainer) -> PintResult<UnitsContainer> {
        let mut result = UnitsContainer::new();
        for (unit_name, &exp) in units.iter() {
            let dim = self.get_unit_dimensionality(unit_name)?;
            result = &result * &dim.pow(exp);
        }
        Ok(result)
    }

    fn ensure_unit_info_cached(&mut self, name: &str) -> PintResult<()> {
        if self.unit_info_cache.contains_key(name) {
            return Ok(());
        }
        let info = self.resolve_unit_ref(name)?;
        self.unit_info_cache.insert(name.to_string(), info);
        Ok(())
    }

    #[inline]
    fn get_unit_dimensionality(&mut self, name: &str) -> PintResult<UnitsContainer> {
        self.ensure_unit_info_cached(name)?;
        Ok(self.unit_info_cache.get(name).unwrap().2.clone())
    }

    /// Get the conversion factor between two unit containers.
    /// If they have different dimensionality, returns an error.
    pub fn get_conversion_factor(
        &mut self,
        src: &UnitsContainer,
        dst: &UnitsContainer,
    ) -> PintResult<f64> {
        // Check dimensionality compatibility
        let src_dim = self.get_dimensionality(src)?;
        let dst_dim = self.get_dimensionality(dst)?;

        if src_dim != dst_dim {
            return Err(Box::new(PintError::DimensionalityError {
                src_units: src.to_string(),
                dst_units: dst.to_string(),
                src_dim: Some(src_dim),
                dst_dim: Some(dst_dim),
            }));
        }

        let src_factor = self.get_root_factor(src)?;
        let dst_factor = self.get_root_factor(dst)?;

        Ok(src_factor / dst_factor)
    }

    /// Get the total conversion factor from a UnitsContainer to root units
    fn get_root_factor(&mut self, units: &UnitsContainer) -> PintResult<f64> {
        let mut factor = 1.0_f64;
        for (unit_name, &exp) in units.iter() {
            let uf = self.get_unit_factor(unit_name)?;
            factor *= uf.powf(exp);
        }
        Ok(factor)
    }

    #[inline]
    fn get_unit_factor(&mut self, name: &str) -> PintResult<f64> {
        self.ensure_unit_info_cached(name)?;
        if let Some(info) = self.unit_info_cache.get(name) {
            return Ok(info.0);
        }
        Err(Box::new(PintError::UndefinedUnitError {
            unit_name: name.to_string(),
        }))
    }

    /// Get the root units (base units) for a UnitsContainer
    pub fn get_root_units(&mut self, units: &UnitsContainer) -> PintResult<(f64, UnitsContainer)> {
        let mut factor = 1.0_f64;
        let mut result = UnitsContainer::new();
        for (unit_name, &exp) in units.iter() {
            let (uf, uru) = self.get_unit_root_units(unit_name)?;
            factor *= uf.powf(exp);
            result = &result * &uru.pow(exp);
        }
        Ok((factor, result))
    }

    #[inline]
    fn get_unit_root_units(&mut self, name: &str) -> PintResult<(f64, UnitsContainer)> {
        self.ensure_unit_info_cached(name)?;
        if let Some(info) = self.unit_info_cache.get(name) {
            return Ok((info.0, info.1.clone()));
        }
        Err(Box::new(PintError::UndefinedUnitError {
            unit_name: name.to_string(),
        }))
    }

    /// Get the offset for a unit
    #[inline]
    pub fn get_unit_offset(&self, name: &str) -> f64 {
        if let Some(canonical) = self.name_map.get(name) {
            if let Some(entry) = self.units.get(canonical) {
                return entry.offset;
            }
        }
        0.0
    }

    /// Check if a unit has a non-zero offset (non-multiplicative)
    #[inline]
    pub fn is_offset_unit(&self, name: &str) -> bool {
        self.get_unit_offset(name) != 0.0
    }

    /// Convert a value from src units to dst units
    pub fn convert(
        &mut self,
        value: f64,
        src: &UnitsContainer,
        dst: &UnitsContainer,
    ) -> PintResult<f64> {
        // Handle offset units
        let src_has_offset = self.has_offset_units(src);
        let dst_has_offset = self.has_offset_units(dst);

        if src_has_offset || dst_has_offset {
            return self.convert_with_offset(value, src, dst);
        }

        let factor = self.get_conversion_factor(src, dst)?;
        Ok(value * factor)
    }

    #[inline]
    fn has_offset_units(&self, units: &UnitsContainer) -> bool {
        for (name, _) in units.iter() {
            if self.is_offset_unit(name) {
                return true;
            }
        }
        false
    }

    fn convert_with_offset(
        &mut self,
        value: f64,
        src: &UnitsContainer,
        dst: &UnitsContainer,
    ) -> PintResult<f64> {
        // For single offset units like degC -> degF:
        // 1. Convert to base (kelvin): val_k = (value + src_offset) * src_factor
        // 2. Convert from base to dst: result = val_k / dst_factor - dst_offset

        let src_dim = self.get_dimensionality(src)?;
        let dst_dim = self.get_dimensionality(dst)?;
        if src_dim != dst_dim {
            return Err(Box::new(PintError::DimensionalityError {
                src_units: src.to_string(),
                dst_units: dst.to_string(),
                src_dim: Some(src_dim),
                dst_dim: Some(dst_dim),
            }));
        }

        // Simple case: single unit on each side
        // Pint's OffsetConverter uses:
        //   to_reference(value)   = value * scale + offset
        //   from_reference(value) = (value - offset) / scale
        if src.len() == 1 && dst.len() == 1 {
            let (src_name, &src_exp) = src.iter().next().unwrap();
            let (dst_name, &dst_exp) = dst.iter().next().unwrap();

            if (src_exp - 1.0).abs() < f64::EPSILON && (dst_exp - 1.0).abs() < f64::EPSILON {
                let src_factor = self.get_unit_factor(src_name)?;
                let dst_factor = self.get_unit_factor(dst_name)?;
                let src_offset = self.get_unit_offset(src_name);
                let dst_offset = self.get_unit_offset(dst_name);

                // to_reference: value * scale + offset -> base unit (kelvin)
                let base_value = value * src_factor + src_offset;
                // from_reference: (base - offset) / scale -> target unit
                let result = (base_value - dst_offset) / dst_factor;
                return Ok(result);
            }
        }

        // Fallback: multiplicative only
        let factor = self.get_conversion_factor(src, dst)?;
        Ok(value * factor)
    }

    /// Check if a unit name is known to the registry
    #[inline]
    pub fn is_known_unit(&self, name: &str) -> bool {
        if self.name_map.contains_key(name) {
            return true;
        }
        self.try_strip_prefix(name).is_some()
    }

    /// Define a new unit at runtime (e.g. from ureg.define())
    pub fn define_unit(&mut self, def: &UnitDef) -> PintResult<()> {
        self.try_add_unit(def)
    }

    /// Define a new prefix at runtime
    pub fn define_prefix(&mut self, p: &PrefixDef) {
        self.add_prefix(p);
    }

    /// Get all unit canonical names that match a given dimensionality
    pub fn get_units_by_dimensionality(&self, dim: &UnitsContainer) -> Vec<String> {
        self.units
            .iter()
            .filter(|(_, entry)| entry.dimensionality == *dim)
            .map(|(name, _)| name.clone())
            .collect()
    }

    /// Get all prefix names with their factors
    pub fn get_all_prefixes(&self) -> Vec<(String, f64)> {
        self.prefixes
            .iter()
            .map(|(name, entry)| (name.clone(), entry.factor))
            .collect()
    }

    /// Get all canonical unit names
    pub fn get_all_unit_names(&self) -> Vec<String> {
        self.units.keys().cloned().collect()
    }

    /// Parse a unit name into (prefix, unit_name, suffix) tuples.
    pub fn parse_prefix_and_unit(&self, name: &str) -> Vec<(String, String, String)> {
        // Direct lookup: no prefix
        if let Some(canonical) = self.name_map.get(name) {
            if self.units.contains_key(canonical) {
                return vec![("".to_string(), canonical.clone(), "".to_string())];
            }
        }
        // Try prefix stripping
        if let Some((_, base_canonical)) = self.try_strip_prefix(name) {
            let prefix_len = name.len() - base_canonical.len();
            if prefix_len > 0 && prefix_len < name.len() {
                let prefix_str = &name[..prefix_len];
                // Find the canonical prefix name
                if let Some(prefix_canonical) = self.prefix_map.get(prefix_str) {
                    return vec![(prefix_canonical.clone(), base_canonical, "".to_string())];
                }
                return vec![(prefix_str.to_string(), base_canonical, "".to_string())];
            }
        }
        vec![]
    }

    /// Get the display name for a unit in a UnitsContainer (prefer symbols)
    pub fn get_display_name(&self, canonical: &str) -> String {
        if let Some(entry) = self.units.get(canonical) {
            if let Some(ref sym) = entry.symbol {
                return sym.clone();
            }
        }
        canonical.to_string()
    }

    /// Format a UnitsContainer using symbols where possible
    pub fn format_units(&self, units: &UnitsContainer) -> String {
        if units.is_empty() {
            return "dimensionless".to_string();
        }

        let mut pos: Vec<_> = units.iter().filter(|(_, &v)| v > 0.0).collect();
        let mut neg: Vec<_> = units.iter().filter(|(_, &v)| v < 0.0).collect();
        pos.sort_by(|a, b| a.0.cmp(b.0));
        neg.sort_by(|a, b| a.0.cmp(b.0));

        let fmt_unit = |name: &str, exp: f64| -> String {
            let display = self.get_display_name(name);
            let abs_exp = exp.abs();
            if (abs_exp - 1.0).abs() < f64::EPSILON {
                display
            } else if abs_exp == abs_exp.floor() {
                format!("{} ** {}", display, abs_exp as i64)
            } else {
                format!("{} ** {}", display, abs_exp)
            }
        };

        let pos_str: Vec<String> = pos.iter().map(|(k, &v)| fmt_unit(k, v)).collect();
        let neg_str: Vec<String> = neg.iter().map(|(k, &v)| fmt_unit(k, v)).collect();

        if neg_str.is_empty() {
            pos_str.join(" * ")
        } else if pos_str.is_empty() {
            format!("1 / {}", neg_str.join(" / "))
        } else {
            format!("{} / {}", pos_str.join(" * "), neg_str.join(" / "))
        }
    }
}

// --- Relation tokenizer ---

#[derive(Debug, Clone, Copy, PartialEq)]
enum RelOp {
    Mul,
    Div,
    Pow,
}

#[derive(Debug, Clone)]
enum RelToken {
    Number(f64),
    UnitRef(String),
    Op(RelOp),
    LParen,
    RParen,
}

fn tokenize_relation(s: &str) -> Vec<RelToken> {
    let mut tokens = Vec::new();
    let chars: Vec<char> = s.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        match chars[i] {
            ' ' | '\t' => i += 1,
            '(' => {
                tokens.push(RelToken::LParen);
                i += 1;
            }
            ')' => {
                tokens.push(RelToken::RParen);
                i += 1;
            }
            '*' => {
                if i + 1 < chars.len() && chars[i + 1] == '*' {
                    tokens.push(RelToken::Op(RelOp::Pow));
                    i += 2;
                } else {
                    tokens.push(RelToken::Op(RelOp::Mul));
                    i += 1;
                }
            }
            '/' => {
                tokens.push(RelToken::Op(RelOp::Div));
                i += 1;
            }
            '^' => {
                tokens.push(RelToken::Op(RelOp::Pow));
                i += 1;
            }
            c if c.is_ascii_digit() || c == '.' || c == '-' => {
                // Check if this could be a negative sign vs minus operator
                let is_unary_minus = c == '-'
                    && (tokens.is_empty()
                        || matches!(tokens.last(), Some(RelToken::Op(_) | RelToken::LParen)));

                if c == '-' && !is_unary_minus {
                    // It's a subtraction operator, treat as a separator
                    tokens.push(RelToken::Op(RelOp::Mul));
                    i += 1;
                    continue;
                }

                let start = i;
                if c == '-' {
                    i += 1;
                }
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
                if let Ok(v) = num_str.parse::<f64>() {
                    tokens.push(RelToken::Number(v));
                }
            }
            _ => {
                // Unit name or dimension reference
                let start = i;
                while i < chars.len() && !is_rel_separator(chars[i]) {
                    i += 1;
                }
                let name: String = chars[start..i].iter().collect();
                let name = name.trim().to_string();
                if !name.is_empty() {
                    tokens.push(RelToken::UnitRef(name));
                }
            }
        }
    }

    tokens
}

fn is_rel_separator(c: char) -> bool {
    matches!(c, ' ' | '\t' | '*' | '/' | '^' | '(' | ')')
}

impl Default for UnitRegistry {
    fn default() -> Self {
        Self::new().expect("Failed to create default UnitRegistry")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_registry() {
        let reg = UnitRegistry::new().unwrap();
        assert!(reg.is_known_unit("meter"));
        assert!(reg.is_known_unit("m"));
        assert!(reg.is_known_unit("second"));
        assert!(reg.is_known_unit("kilogram"));
    }

    #[test]
    fn test_prefixed_units() {
        let reg = UnitRegistry::new().unwrap();
        assert!(reg.is_known_unit("kilometer"));
        assert!(reg.is_known_unit("millisecond"));
        assert!(reg.is_known_unit("microgram"));
    }

    #[test]
    fn test_conversion() {
        let reg = UnitRegistry::new().unwrap();
        let km = UnitsContainer::from_single("kilometer".to_string(), 1.0);
        let m = UnitsContainer::from_single("meter".to_string(), 1.0);
        let factor = reg.get_conversion_factor(&km, &m).unwrap();
        assert!((factor - 1000.0).abs() < 1e-10);
    }

    #[test]
    fn test_dimensionality_check() {
        let reg = UnitRegistry::new().unwrap();
        let m = UnitsContainer::from_single("meter".to_string(), 1.0);
        let s = UnitsContainer::from_single("second".to_string(), 1.0);
        assert!(reg.get_conversion_factor(&m, &s).is_err());
    }

    #[test]
    fn test_meter_dimensionality() {
        let reg = UnitRegistry::new().unwrap();
        let m = UnitsContainer::from_single("meter".to_string(), 1.0);
        let dim = reg.get_dimensionality(&m).unwrap();
        assert!(!dim.is_empty());
        assert_eq!(dim.get("[length]"), 1.0);
    }
}
