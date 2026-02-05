use thiserror::Error;

use crate::http_client;
use crate::wasm_compat::WasmCompatSend;

#[derive(Debug, Error)]
pub enum VerifyError {
	#[error("invalid authentication")]
	InvalidAuthentication,
	#[error("provider error: {0}")]
	ProviderError(String),
	#[error("http error: {0}")]
	HttpError(
		#[from]
		#[source]
		http_client::Error,
	),
}

/// A provider client that can verify the configuration.
/// Clone is required for conversions between client types.
pub trait VerifyClient {
	/// Verify the configuration.
	fn verify(&self) -> impl Future<Output = Result<(), VerifyError>> + WasmCompatSend;
}
