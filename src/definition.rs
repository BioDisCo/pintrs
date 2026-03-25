/// Parsed unit/prefix/dimension definitions from the definition files.

#[derive(Debug, Clone)]
pub struct PrefixDef {
    pub name: String,
    pub factor: f64,
    pub symbol: Option<String>,
    pub aliases: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct UnitDef {
    pub name: String,
    /// The raw relation string, e.g. "1e-10 * meter" or "[length]"
    pub relation: String,
    pub symbol: Option<String>,
    pub aliases: Vec<String>,
    /// If this is a base dimension, e.g. meter = [length]
    pub is_base: bool,
    pub dimension: Option<String>,
    /// Offset for non-multiplicative units like degC
    pub offset: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct DimensionDef {
    pub name: String,
    /// The relation string, e.g. "[mass] / [volume]"
    pub relation: Option<String>,
}

#[derive(Debug, Clone)]
pub struct AliasDef {
    pub name: String,
    pub aliases: Vec<String>,
}

#[derive(Debug, Clone)]
pub enum Definition {
    Prefix(PrefixDef),
    Unit(UnitDef),
    Dimension(DimensionDef),
    Alias(AliasDef),
    Defaults(()),
    Comment,
    Import(String),
}
