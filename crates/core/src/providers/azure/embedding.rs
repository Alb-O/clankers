use serde::Deserialize;
use serde_json::json;

use super::client::Client;
use crate::completion::GetTokenUsage;
use crate::embeddings::{self, EmbeddingError};
use crate::http_client::{self, HttpClientExt};
use crate::providers::openai_compat::ApiResponse;

fn model_dimensions_from_identifier(identifier: &str) -> Option<usize> {
	match identifier {
		super::TEXT_EMBEDDING_3_LARGE => Some(3_072),
		super::TEXT_EMBEDDING_3_SMALL | super::TEXT_EMBEDDING_ADA_002 => Some(1_536),
		_ => None,
	}
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingResponse {
	pub object: String,
	pub data: Vec<EmbeddingData>,
	pub model: String,
	pub usage: Usage,
}

impl From<ApiResponse<EmbeddingResponse>> for Result<EmbeddingResponse, EmbeddingError> {
	fn from(value: ApiResponse<EmbeddingResponse>) -> Self {
		match value {
			ApiResponse::Ok(response) => Ok(response),
			ApiResponse::Err(err) => Err(EmbeddingError::ProviderError(err.message)),
		}
	}
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingData {
	pub object: String,
	pub embedding: Vec<f64>,
	pub index: usize,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Usage {
	pub prompt_tokens: usize,
	pub total_tokens: usize,
}

impl GetTokenUsage for Usage {
	fn token_usage(&self) -> Option<crate::completion::Usage> {
		let mut usage = crate::completion::Usage::new();

		usage.input_tokens = self.prompt_tokens as u64;
		usage.total_tokens = self.total_tokens as u64;
		usage.output_tokens = usage.total_tokens - usage.input_tokens;

		Some(usage)
	}
}

impl std::fmt::Display for Usage {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		write!(
			f,
			"Prompt tokens: {} Total tokens: {}",
			self.prompt_tokens, self.total_tokens
		)
	}
}

#[derive(Clone)]
pub struct EmbeddingModel<T = reqwest::Client> {
	client: Client<T>,
	pub model: String,
	ndims: usize,
}

impl<T> embeddings::EmbeddingModel for EmbeddingModel<T>
where
	T: HttpClientExt + Default + Clone + 'static,
{
	const MAX_DOCUMENTS: usize = 1024;

	type Client = Client<T>;

	fn make(client: &Self::Client, model: impl Into<String>, dims: Option<usize>) -> Self {
		Self::new(client.clone(), model, dims)
	}

	fn ndims(&self) -> usize {
		self.ndims
	}

	async fn embed_texts(
		&self,
		documents: impl IntoIterator<Item = String>,
	) -> Result<Vec<embeddings::Embedding>, EmbeddingError> {
		let documents = documents.into_iter().collect::<Vec<_>>();

		let mut body = json!({
			"input": documents,
		});

		if self.ndims > 0 && self.model.as_str() != super::TEXT_EMBEDDING_ADA_002 {
			body["dimensions"] = json!(self.ndims);
		}

		let body = serde_json::to_vec(&body)?;

		let req = self
			.client
			.post_embedding(self.model.as_str())?
			.body(body)
			.map_err(|e| EmbeddingError::HttpError(e.into()))?;

		let response = self.client.send(req).await?;

		if response.status().is_success() {
			let body: Vec<u8> = response.into_body().await?;
			let body: ApiResponse<EmbeddingResponse> = serde_json::from_slice(&body)?;

			match body {
				ApiResponse::Ok(response) => {
					tracing::info!(target: "clankers",
						"Azure embedding token usage: {}",
						response.usage
					);

					if response.data.len() != documents.len() {
						return Err(EmbeddingError::ResponseError(
							"Response data length does not match input length".into(),
						));
					}

					Ok(response
						.data
						.into_iter()
						.zip(documents.into_iter())
						.map(|(embedding, document)| embeddings::Embedding {
							document,
							vec: embedding.embedding,
						})
						.collect())
				}
				ApiResponse::Err(err) => Err(EmbeddingError::ProviderError(err.message)),
			}
		} else {
			let text = http_client::text(response).await?;
			Err(EmbeddingError::ProviderError(text))
		}
	}
}

impl<T> EmbeddingModel<T> {
	pub fn new(client: Client<T>, model: impl Into<String>, ndims: Option<usize>) -> Self {
		let model = model.into();
		let ndims = ndims
			.or(model_dimensions_from_identifier(&model))
			.unwrap_or_default();

		Self {
			client,
			model,
			ndims,
		}
	}

	pub fn with_model(client: Client<T>, model: &str, ndims: Option<usize>) -> Self {
		let ndims = ndims.unwrap_or_default();

		Self {
			client,
			model: model.into(),
			ndims,
		}
	}
}
