use axum::{
    extract::{Json, State},
    response::IntoResponse,
    http::StatusCode,
    routing::{post},
    Router, Extension,
};
use serde::{Deserialize, Serialize};
use std::{sync::Arc, path::PathBuf};
use std::any::Any;
use std::iter::Sum;
use std::net::SocketAddr;
use tokio::sync::Mutex;
use crate::chat::ChatAI;
use crate::config::{LlamaConfigJson, TorchDType};
use crate::model;
use half::{bf16, f16};
use num_traits::{Float, FromPrimitive};
use crate::params::FromLeBytes;
use log::info;

/// **请求结构体**
#[derive(Deserialize)]
struct ChatRequest {
    user_input: String,
}

/// **响应结构体**
#[derive(Serialize)]
struct ChatResponse {
    response: String,
}

/// **状态管理**
struct AppState<T> {
    chat_ai: Mutex<ChatAI<T>>, // 线程安全
}

impl<T> AppState<T>
where
    T: Float + Default + Copy + Sum + FromPrimitive + FromLeBytes,
{
    fn new(model_dir: PathBuf, config: LlamaConfigJson) -> Arc<Self> {
        let chat_ai = ChatAI::<T>::new(model_dir, config);
        Arc::new(AppState {
            chat_ai: Mutex::new(chat_ai),
        })
    }
}

/// **聊天 API**
async fn chat_api<T>(
    State(state): State<Arc<AppState<T>>>,
    Json(payload): Json<ChatRequest>,
) -> impl IntoResponse
where
    T: Float + Default + Copy + Sum + FromPrimitive + FromLeBytes + Send + Sync + 'static,
{
    let mut chat_ai = state.chat_ai.lock().await;
    let response_text = chat_ai.chat(&payload.user_input);

    Json(ChatResponse { response: response_text })
}

/// **启动 Web 服务器**
pub async fn start_server() {
    info!("Starting AI Server...");

    // 读取模型配置
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("chat");
    let config = model::read_config(&model_dir);

    // 选择正确的 ChatAI 实例
    let chat_ai_state: Arc<dyn Any + Send + Sync> = match config.torch_dtype {
        TorchDType::Float32 => AppState::<f32>::new(model_dir.clone(), config.clone()),
        TorchDType::Float16 => AppState::<f16>::new(model_dir.clone(), config.clone()),
        TorchDType::BFloat16 => AppState::<bf16>::new(model_dir.clone(), config.clone()),
    };

    let chat_ai_extension = match config.torch_dtype {
        TorchDType::Float32 => {
            let state = AppState::<f32>::new(model_dir.clone(), config.clone());
            Router::new().route("/chat", post(chat_api::<f32>)).with_state(state)
        }
        TorchDType::Float16 => {
            let state = AppState::<f16>::new(model_dir.clone(), config.clone());
            Router::new().route("/chat", post(chat_api::<f16>)).with_state(state)
        }
        TorchDType::BFloat16 => {
            let state = AppState::<bf16>::new(model_dir.clone(), config.clone());
            Router::new().route("/chat", post(chat_api::<bf16>)).with_state(state)
        }
    };

    let app = Router::new()
        .merge(chat_ai_extension)
        .route_layer(Extension(config));

    // ✅ 显式指定 `SocketAddr`
    let addr: SocketAddr = "127.0.0.1:8000".parse().unwrap();
    info!("🚀 Server running at {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app.into_make_service_with_connect_info::<SocketAddr>())
        .await
        .unwrap();
}