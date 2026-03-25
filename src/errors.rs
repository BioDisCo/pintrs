use std::fmt;

use crate::units_container::UnitsContainer;

#[derive(Debug, Clone)]
pub enum PintError {
    DimensionalityError {
        src_units: String,
        dst_units: String,
        src_dim: Option<UnitsContainer>,
        dst_dim: Option<UnitsContainer>,
    },
    UndefinedUnitError {
        unit_name: String,
    },
    OffsetUnitCalculusError {
        src_units: String,
        dst_units: String,
    },
    DefinitionSyntaxError {
        message: String,
    },
    RedefinitionError {
        name: String,
    },
}

impl fmt::Display for PintError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PintError::DimensionalityError {
                src_units,
                dst_units,
                src_dim,
                dst_dim,
            } => {
                write!(
                    f,
                    "Cannot convert from '{}' ({}) to '{}' ({})",
                    src_units,
                    src_dim.as_ref().map(|d| d.to_string()).unwrap_or_default(),
                    dst_units,
                    dst_dim.as_ref().map(|d| d.to_string()).unwrap_or_default(),
                )
            }
            PintError::UndefinedUnitError { unit_name } => {
                write!(f, "'{}' is not defined in the unit registry", unit_name)
            }
            PintError::OffsetUnitCalculusError {
                src_units,
                dst_units,
            } => {
                write!(
                    f,
                    "Ambiguous operation with offset unit ({}, {})",
                    src_units, dst_units
                )
            }
            PintError::DefinitionSyntaxError { message } => {
                write!(f, "Definition syntax error: {}", message)
            }
            PintError::RedefinitionError { name } => {
                write!(f, "Cannot redefine '{}'", name)
            }
        }
    }
}

impl std::error::Error for PintError {}

pub type PintResult<T> = Result<T, PintError>;
