use axum::{extract::{Json, State}, response::IntoResponse, routing::{post}, Router, Extension, body::Body, ServiceExt};
use futures_util::stream::StreamExt;
use serde::{Deserialize, Serialize};
use std::{sync::Arc, path::PathBuf, net::SocketAddr, pin::Pin, io};
use std::any::Any;
use std::iter::Sum;
use std::ops::{Deref, DerefMut};
use tokio::sync::{Mutex, mpsc, RwLock};
use tokio_stream::wrappers::ReceiverStream;
use crate::chat::{ChatAI, Message};
use crate::config::{LlamaConfigJson, TorchDType};
use crate::model;
use half::{bf16, f16};
use num_traits::{Float, FromPrimitive};
use crate::params::FromLeBytes;
use log::info;
use tokio::io::AsyncWriteExt;
use tokio::net::TcpListener;
use crate::kvcache::{KVCache, KVCacheManager};


// curl -N -X POST http://127.0.0.1:8000/chat \
//      -H "Content-Type: application/json" \
//      -d '{"user_id": "user1", "user_input": "Hello"}'

/// **请求结构体**
#[derive(Deserialize)]
struct ChatRequest {
    user_id: String,  // ✅ 允许传入 `user_id`，用于区分缓存
    user_input: String,
}

/// **状态管理**
struct AppState<T> {
    chat_ai: Arc<ChatAI<T>>,  // ✅ 变成 `Arc<ChatAI<T>>`，只读
    kvcache: Arc<KVCacheManager<T>>,  // ✅ `KVCacheManager` 变成 `Arc`，支持并发
}

impl<T> AppState<T>
where
    T: Float + Default + Copy + Sum + FromPrimitive + FromLeBytes + std::marker::Send + std::marker::Sync + 'static,
{
    fn new(model_dir: PathBuf, config: LlamaConfigJson) -> Arc<Self> {
        let chat_ai = Arc::new(ChatAI::<T>::new(model_dir, config));
        let kvcache = KVCacheManager::new(chat_ai.llama.n_layers, chat_ai.llama.max_seq_len, chat_ai.llama.dqkv * chat_ai.llama.n_kv_h);
        Arc::new(AppState { chat_ai, kvcache })
    }
}

/// **流式聊天 API**
async fn chat_api_stream<T>(
    State(state): State<Arc<AppState<T>>>,
    Json(payload): Json<ChatRequest>,
) -> impl IntoResponse
where
    T: Float + Default + Copy + Sum + FromPrimitive + FromLeBytes + Send + Sync + 'static,
{
    let (tx, rx) = mpsc::channel::<String>(10);
    let user_id = payload.user_id.clone();
    let user_input = payload.user_input.clone();
    let state = state.clone();

    tokio::spawn(async move {
        let chat_ai = &state.chat_ai;  // ✅ 只读，不加锁

        // ✅ **1. 获取用户 `KVCache<T>`（只读访问 `Arc<KVCache<T>>`）**
        let user_kvcache = state.kvcache.get_cache_for_user(&user_id).await;

        // ✅ **2. 创建 `KVCacheEntry` 副本**
        let mut binding = user_kvcache.write().await;

        // ✅ **3. 处理对话逻辑**
        let conversation_input = {
            let mut messages = chat_ai.messages.clone();
            messages.push(Message {
                role: "user".to_string(),
                content: user_input.clone(),
            });
            let mut prompt = String::new();
            messages.first().map(|msg| prompt.push_str(&msg.format()));
            prompt.push_str("<|im_start|>assistant\n");
            prompt
        };

        // ✅ **4. 编码 `input_ids`**
        let encoded = chat_ai.tokenizer.encode(conversation_input, true).unwrap();
        let input_ids = encoded.get_ids().to_vec();

        // ✅ **5. 生成 Token（流式返回）**
        let response_tokens = chat_ai.llama.streaming_generate(
            &input_ids,
            128,
            T::from(0.7).unwrap(),
            30,
            T::from(1.0).unwrap(),
            binding.deref_mut(),  // ✅ **传递 `KVCacheEntry` 副本**
        );

        let mut response_text = String::new();
        for token in response_tokens {
            let token_str = chat_ai.tokenizer.decode(&[token], true).unwrap() + " ";  // ✅ **Token 之间加空格**
            response_text.push_str(&token_str);
            // info!("{}", token_str);

            // ✅ **流式返回**
            if tx.send(token_str.clone()).await.is_err() {
                break;
            }

            // ✅ **立即刷新，确保客户端立刻收到**
            tokio::task::yield_now().await;  // ✅ 让 tokio 任务切换，提高响应速度
            // ✅ **遇到 `eos_token_id` 直接停止**
            if token == chat_ai.llama.eos_token_id {
                break;
            }
        }
        info!("response_text: {}", response_text);

        // ✅ 5. 存储用户的 KVCache
        // state.kvcache.store_cache_for_user(&user_id, user_kvcache).await;
    });

    let body_stream = ReceiverStream::new(rx).map(|chunk| Ok::<_, std::io::Error>(chunk));
    Body::from_stream(body_stream)
}





/// **启动 Web 服务器**
pub async fn start_server() {
    info!("Starting AI Server...");

    // 读取模型配置
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("chat");
    let config = model::read_config(&model_dir);

    // 选择正确的 ChatAI 实例
    let chat_ai_extension = match config.torch_dtype {
        TorchDType::Float32 => {
            let state = AppState::<f32>::new(model_dir.clone(), config.clone());
            Router::new().route("/chat", post(chat_api_stream::<f32>)).with_state(state)
        }
        TorchDType::Float16 => {
            let state = AppState::<f16>::new(model_dir.clone(), config.clone());
            Router::new().route("/chat", post(chat_api_stream::<f16>)).with_state(state)
        }
        TorchDType::BFloat16 => {
            let state = AppState::<bf16>::new(model_dir.clone(), config.clone());
            Router::new().route("/chat", post(chat_api_stream::<bf16>)).with_state(state)
        }
    };

    let app = Router::new()
        .merge(chat_ai_extension)
        .route_layer(Extension(config));

    let addr: SocketAddr = "127.0.0.1:8000".parse().unwrap();
    info!("🚀 Server running at {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app.into_make_service_with_connect_info::<SocketAddr>())
        .await
        .unwrap();
}