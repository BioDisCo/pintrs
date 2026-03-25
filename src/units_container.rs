use indexmap::IndexMap;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::ops::{Div, Mul};

/// Stores the product of unit names and their exponents.
/// e.g. {"meter": 1, "second": -2} represents m/s^2
#[derive(Clone, Debug)]
pub struct UnitsContainer {
    inner: IndexMap<String, f64>,
}

impl UnitsContainer {
    #[inline]
    pub fn new() -> Self {
        Self {
            inner: IndexMap::new(),
        }
    }

    pub fn from_map(map: IndexMap<String, f64>) -> Self {
        let mut uc = Self {
            inner: IndexMap::new(),
        };
        for (k, v) in map {
            if v != 0.0 {
                uc.inner.insert(k, v);
            }
        }
        uc
    }

    #[inline]
    pub fn from_single(name: String, exp: f64) -> Self {
        let mut inner = IndexMap::new();
        if exp != 0.0 {
            inner.insert(name, exp);
        }
        Self { inner }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    #[inline]
    pub fn get(&self, key: &str) -> f64 {
        self.inner.get(key).copied().unwrap_or(0.0)
    }

    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = (&String, &f64)> {
        self.inner.iter()
    }

    #[inline]
    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.inner.keys()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    #[inline]
    pub fn contains(&self, key: &str) -> bool {
        self.inner.contains_key(key)
    }

    /// Add an exponent to an existing key, or insert if not present.
    /// Removes the key if the resulting exponent is zero.
    pub fn add(&self, key: &str, value: f64) -> Self {
        let mut new = self.clone();
        let current = new.inner.get(key).copied().unwrap_or(0.0);
        let result = current + value;
        if result == 0.0 {
            new.inner.shift_remove(key);
        } else {
            new.inner.insert(key.to_string(), result);
        }
        new
    }

    pub fn rename(&self, old: &str, new_name: &str) -> Self {
        let mut result = self.clone();
        if let Some(val) = result.inner.shift_remove(old) {
            result.inner.insert(new_name.to_string(), val);
        }
        result
    }

    pub fn inner_map(&self) -> &IndexMap<String, f64> {
        &self.inner
    }
}

impl Default for UnitsContainer {
    fn default() -> Self {
        Self::new()
    }
}

impl PartialEq for UnitsContainer {
    fn eq(&self, other: &Self) -> bool {
        if self.inner.len() != other.inner.len() {
            return false;
        }
        for (k, v) in &self.inner {
            match other.inner.get(k) {
                Some(ov) if (v - ov).abs() < f64::EPSILON => {}
                _ => return false,
            }
        }
        true
    }
}

impl Eq for UnitsContainer {}

impl Hash for UnitsContainer {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let mut pairs: Vec<_> = self.inner.iter().collect();
        pairs.sort_by(|a, b| a.0.cmp(b.0));
        for (k, v) in pairs {
            k.hash(state);
            v.to_bits().hash(state);
        }
    }
}

impl Mul for &UnitsContainer {
    type Output = UnitsContainer;

    fn mul(self, rhs: Self) -> UnitsContainer {
        let mut result = self.inner.clone();
        for (k, v) in &rhs.inner {
            let entry = result.entry(k.clone()).or_insert(0.0);
            *entry += v;
            if *entry == 0.0 {
                result.shift_remove(k);
            }
        }
        UnitsContainer { inner: result }
    }
}

impl Div for &UnitsContainer {
    type Output = UnitsContainer;

    fn div(self, rhs: Self) -> UnitsContainer {
        let mut result = self.inner.clone();
        for (k, v) in &rhs.inner {
            let entry = result.entry(k.clone()).or_insert(0.0);
            *entry -= v;
            if *entry == 0.0 {
                result.shift_remove(k);
            }
        }
        UnitsContainer { inner: result }
    }
}

impl UnitsContainer {
    /// Raise all exponents to a power
    pub fn pow(&self, power: f64) -> Self {
        let mut inner = IndexMap::new();
        for (k, v) in &self.inner {
            let new_exp = v * power;
            if new_exp != 0.0 {
                inner.insert(k.clone(), new_exp);
            }
        }
        Self { inner }
    }

    /// Invert: negate all exponents
    pub fn inv(&self) -> Self {
        self.pow(-1.0)
    }
}

impl fmt::Display for UnitsContainer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.inner.is_empty() {
            return write!(f, "dimensionless");
        }

        let mut pos: Vec<_> = self.inner.iter().filter(|(_, &v)| v > 0.0).collect();
        let mut neg: Vec<_> = self.inner.iter().filter(|(_, &v)| v < 0.0).collect();
        pos.sort_by(|a, b| a.0.cmp(b.0));
        neg.sort_by(|a, b| a.0.cmp(b.0));

        let fmt_part = |items: &[(&String, &f64)]| -> String {
            items
                .iter()
                .map(|(k, &v)| {
                    let abs_v = v.abs();
                    if (abs_v - 1.0).abs() < f64::EPSILON {
                        k.to_string()
                    } else if abs_v == abs_v.floor() {
                        format!("{} ** {}", k, abs_v as i64)
                    } else {
                        format!("{} ** {}", k, abs_v)
                    }
                })
                .collect::<Vec<_>>()
                .join(" * ")
        };

        if neg.is_empty() {
            write!(f, "{}", fmt_part(&pos))
        } else if pos.is_empty() {
            write!(f, "1 / {}", fmt_part(&neg))
        } else {
            write!(f, "{} / {}", fmt_part(&pos), fmt_part(&neg))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mul_div() {
        let a = UnitsContainer::from_single("meter".into(), 1.0);
        let b = UnitsContainer::from_single("second".into(), 1.0);
        let result = &a * &b;
        assert_eq!(result.get("meter"), 1.0);
        assert_eq!(result.get("second"), 1.0);

        let result = &a / &b;
        assert_eq!(result.get("meter"), 1.0);
        assert_eq!(result.get("second"), -1.0);
    }

    #[test]
    fn test_cancellation() {
        let a = UnitsContainer::from_single("meter".into(), 1.0);
        let b = UnitsContainer::from_single("meter".into(), 1.0);
        let result = &a / &b;
        assert!(result.is_empty());
    }

    #[test]
    fn test_display() {
        let uc = UnitsContainer::from_single("meter".into(), 1.0);
        assert_eq!(uc.to_string(), "meter");
    }
}
