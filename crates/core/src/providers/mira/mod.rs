//! Mira API client and Clankers integration
//!
//! # Example
//! ```
//! use clankers::providers::mira;
//!
//! let client = mira::Client::new("YOUR_API_KEY");
//!
//! ```

pub mod client;
pub mod completion;

pub use client::{Client, ClientBuilder};
pub use completion::{CompletionModel, CompletionResponse};
