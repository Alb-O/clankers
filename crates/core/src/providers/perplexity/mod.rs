//! Perplexity API client and Clankers integration
//!
//! # Example
//! ```
//! use clankers::providers::perplexity;
//!
//! let client = perplexity::Client::new("YOUR_API_KEY");
//!
//! let sonar = client.completion_model(perplexity::SONAR);
//! ```

pub mod client;
pub mod completion;

pub use client::{Client, ClientBuilder};
pub use completion::{SONAR, SONAR_PRO};
