pub mod client;
pub mod completion;
pub mod embedding;

pub use client::{Client, ClientBuilder};
pub use completion::{
	CODESTRAL, CODESTRAL_MAMBA, CompletionModel, MINISTRAL_3B, MINISTRAL_8B, MISTRAL_LARGE,
	MISTRAL_NEMO, MISTRAL_SABA, MISTRAL_SMALL, PIXTRAL_LARGE, PIXTRAL_SMALL,
};
pub use embedding::{EmbeddingModel, MISTRAL_EMBED};
