
import asyncio
from datetime import datetime, timedelta
import json
import logging
import os
import traceback
import uuid
import base64
from typing import Dict, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ValidationError

# from model import cosyvoice_model
import model
from utils import convert_to_format, download_audio_file, cleanup_temp_file, ResponseBuilder, AudioProcessingError, validate_message, ErrorCode
from cosyvoice.utils.file_utils import load_wav

logger = logging.getLogger(__name__)
router = APIRouter()

class WebSocketMessage(BaseModel):
    header: dict
    payload: dict
    
class HeaderMessage(BaseModel):
    version: str
    message_type: str
    timestamp: str
    sequence: int

class TTSParamsMessage(BaseModel):
    text: str
    mode: str = "zero_shot"
    prompt_audio_path: str = "./asset/zero_shot_prompt.wav"
    prompt_text: str = "希望你以后能够做的比我还好呦。"
    instruct_text: Optional[str] = ""
    speed: float = 1.0
    output_format: str = "wav"  # pcm, wav, mp3
    zero_shot_spk_id: str = ''
    
    # 音频编码相关参数
    bit_rate: Optional[int] = 192000        # 比特率，默认 192kbps
    compression_level: Optional[int] = 2     # 压缩级别 (0-9)
    
class ConnectionManager:
    def __init__(self, max_connections=100, timeout_minutes: int = None):
        self.active_connections: Dict[WebSocket, datetime] = {} #  存储 connection 和最后的活跃时间
        self.max_connections = max_connections
        self.timeout_minutes = timeout_minutes or int(os.getenv('CONNECTION_TIMEOUT_MINUTES', 30)) # websocket 的超时时间，从环境变量里面拿
        self.lock = asyncio.Lock()  # 异步锁，保证多个 client 能同时连接或断开，针对并发场景
        
    async def connect(self, websocket: WebSocket) -> bool:
        async with self.lock:
            if len(self.active_connections) >= self.max_connections:
                logger.warning("Maximum connection limit reached")
                return False
            
            await websocket.accept()
            self.active_connections[websocket] = datetime.now()
            logger.info("New connection added successfully!")
            return True
    
    async def disconnect(self, websocket: WebSocket):
        async with self.lock:
            if websocket in self.active_connections:
                del self.active_connections[websocket]
                logger.info("WebSocket removed successfully!")
    
    def get_connection_count(self) -> int:
        return len(self.active_connections)
    
    # connection 超时时间相关
    async def update_connection_activity(self, websocket: WebSocket):
        """更新 connection 活跃时间"""
        async with self.lock:
            if websocket in self.active_connections:
                self.active_connections[websocket] = datetime.now()

    async def check_connection_timeout(self, websocket: WebSocket) -> bool:
        """检查是否超时"""
        if websocket not in self.active_connections:
            return True

        last_active = self.active_connections[websocket]
        return datetime.now() - last_active > timedelta(minutes=self.timeout_minutes)

    async def cleanup_timeout_connections(self):
        """清理超时 connection"""
        async with self.lock:
            now = datetime.now()
            timeout_connections = [
                ws for ws, last_active in self.active_connections.items()
                if now - last_active > timedelta(minutes=self.timeout_minutes)
            ]
            
            for ws in timeout_connections:
                try:
                    await ws.close(code=1000, reason="Connection timeout")
                    del self.active_connections[ws]
                    logger.info(f"Closed timeout connection")
                except Exception as e:
                    logger.error(f"Error closing timeout connection: {e}")
    
class SessionManager:
    def __init__(self, max_sessions=50, timeout_minutes: int = None):
        self.sessions: Dict[str, dict] = {}
        self.timeout_minutes = timeout_minutes or int(os.getenv('SESSION_TIMEOUT_MINUTES', 30)) # session 的超时时间，从环境变量里面获取
        self.max_sessions = max_sessions
        self.lock = asyncio.Lock()  # 同 websocket
        
    async def create_session(self, websocket: WebSocket) -> str:   
        async with self.lock:
            if len(self.sessions) >= self.max_sessions:
                raise ValueError(f"Maximum session limit ({self.max_sessions}) reached")
            
            session_id = str(uuid.uuid4())
            self.sessions[session_id] = {
                "websocket": websocket,
                "created_at": datetime.now(),
                "status": "active"
            }
            logger.info("Create session successfully!")
            return session_id

    async def close_session(self, session_id: str):
        """关闭并清理session"""
        async with self.lock:
            if session_id in self.sessions:
                try:
                    session = self.sessions[session_id]
                    del self.sessions[session_id]
                    logger.info(f"Session {session_id} removed successfully")
                    
                except Exception as e:
                    logger.error(f"Error while closing session {session_id}: {e}")
                    raise

    async def get_session(self, session_id: str) -> Optional[dict]:
        return self.sessions.get(session_id)
    
class TTSHandler:
    def __init__(self, connection_manager: ConnectionManager, session_manager: SessionManager):
        self.connection_manager = connection_manager
        self.session_manager = session_manager
        self.supported_versions = ["1.0"]
        self.response_builder = ResponseBuilder()
        self.pending_tasks = {} # 存储每个 session 的待处理任务
        self.audio_tasks_status = {}
    
    # request: CONNECT_REQUEST / response: CONNECT_RESPONSE
    async def handle_connect_request(self, websocket: WebSocket, message: dict) -> dict:
        try:
            is_valid, error = validate_message(message, self.supported_versions)
            if not is_valid:
                return self.response_builder.create_error_response(
                    message,
                    ErrorCode.INVALID_PARAMS,
                    "Invalid message",
                    error
                )
            return self.response_builder.create_response(
                    message=message,
                    message_type="CONNECT_RESPONSE",
                    payload={
                        "server_version": "1.0.0",
                        "server_status": "ready"
                    }
                )
            
        except Exception as e:
            logger.error(f"Handling connect request faild due to: {str(e)}")
            logger.error(traceback.format_exc())
            return self.response_builder.create_error_response(
                message,
                ErrorCode.INTERNAL_ERROR,
                "Internal server error, please try again.",
                str(e)
            )
    # request: SESSION_REQUEST / response: SESSION_RESPONSE
    async def handle_session_request(self, websocket: WebSocket, message: dict) -> dict:
        try:
            is_valid, error = validate_message(message, self.supported_versions)
            if not is_valid:
                return self.response_builder.create_error_response(
                    message,
                    ErrorCode.INVALID_PARAMS,
                    "Invalid message",
                    error
                )

            existing_sessions = []
            async with self.session_manager.lock:
                for session_id, session_data in self.session_manager.sessions.items():
                    if session_data["websocket"] == websocket:
                        existing_sessions.append(session_id)

            for session_id in existing_sessions:
                logger.info(f"Cleaning up existing session {session_id} for websocket")

                # 清理任务
                if session_id in self.pending_tasks:
                    for task in self.pending_tasks[session_id]:
                        if not task.done():
                            task.cancel()
                            try:
                                await task
                            except asyncio.CancelledError:
                                pass
                            except Exception as e:
                                logger.error(f"Error canceling task for session {session_id}: {e}")
                    del self.pending_tasks[session_id]
                
                # 清理状态
                if session_id in self.audio_tasks_status:
                    del self.audio_tasks_status[session_id]
                
                # 关闭会话
                await self.session_manager.close_session(session_id)
                
            logger.info(f"Current sessions: {len(self.session_manager.sessions)}")
            logger.info(f"Current pending tasks: {len(self.pending_tasks)}")
            logger.info(f"Current audio tasks: {len(self.audio_tasks_status)}")

            # 创建新会话
            try:
                session_id = await self.session_manager.create_session(websocket)
                logger.info(f"Created new session {session_id}")
                return self.response_builder.create_response(
                    message=message,
                    message_type="SESSION_RESPONSE",
                    payload={
                        "session_id": session_id
                    }
                )
            except ValueError as e:
                return self.response_builder.create_error_response(
                    message,
                    ErrorCode.RESOURCE_EXHAUSTED,
                    "Reached max session counts, please try it later.",
                    str(e)
                )
                
        except Exception as e:
            logger.error(f"Handling session request failed due to: {str(e)}")
            logger.error(traceback.format_exc())
            return self.response_builder.create_error_response(
                message,
                ErrorCode.INTERNAL_ERROR,
                "Internal server error, please try again.",
                str(e)
            )

    # request: SYNTHESIS_START / response: SYNTHESIS_START_ACK
    async def handle_synthesis_start(self, websocket: WebSocket, message: dict) -> dict:
        try:
            is_valid, error = validate_message(message, self.supported_versions)
            if not is_valid:
                return self.response_builder.create_error_response(
                    message,
                    ErrorCode.INVALID_PARAMS,
                    "Invalid message",
                    error
                )
            session_id = message["payload"].get("session_id")
            if not session_id:
                return self.response_builder.create_error_response(
                    message,
                    ErrorCode.INVALID_SESSION,
                    "Invalid session"
                )
            return self.response_builder.create_response(
                message=message,
                message_type="SYNTHESIS_START_ACK",
                payload={
                    "session_id": session_id,
                    "status": "ready"
                }
            )
        except Exception as e:
            logger.error(f"Failed to handle synthesis start request due to: {str(e)}")
            logger.error(traceback.format_exc())
            return self.response_builder.create_error_response(
                message,
                ErrorCode.INTERNAL_ERROR,
                "Internal server error, please try again.",
                str(e)
            )
            
    # request: TEXT_REQUEST / response: AUDIO_RESPONSE       
    async def handle_text_request(self, websocket: WebSocket, message: dict) -> dict:
        temp_file = None
        try:
            is_valid, error = validate_message(message, self.supported_versions)
            if not is_valid:
                return self.response_builder.create_error_response(
                    message,
                    ErrorCode.INVALID_PARAMS,
                    "Invalid message",
                    error
                )
                
            session_id = message["payload"].get("session_id")
            if not session_id:
                return self.response_builder.create_error_response(
                    message,
                    ErrorCode.INVALID_SESSION,
                    "Invalid session"
                )
            if session_id not in self.pending_tasks:
                self.pending_tasks[session_id] = []
            if session_id not in self.audio_tasks_status:
                self.audio_tasks_status[session_id] = {
                    "received_texts": 0,
                    "completed_texts": 0
                }
            
            self.audio_tasks_status[session_id]["received_texts"] += 1
            current_text_index = self.audio_tasks_status[session_id]["received_texts"]
                
            try:
                params = TTSParamsMessage(**message["payload"]["params"])
            except ValidationError as e:
                return self.response_builder.create_error_response(
                    message,
                    ErrorCode.INVALID_PARAMS,
                    "Invalid TTS request params",
                    str(e)
                )
                
            try:
                if params.prompt_audio_path.startswith(('http://', 'https://')):
                    temp_file = await download_audio_file(params.prompt_audio_path)
                    prompt_speech = load_wav(temp_file, 16000)
                else:
                    prompt_speech = load_wav(params.prompt_audio_path, 16000)
            except Exception as e:
                return self.response_builder.create_error_response(
                    message,
                    ErrorCode.INVALID_PARAMS,
                    "Failed to load prompt audio",
                    str(e)
                )
                
            try:
                if params.mode == "zero_shot":
                    inference_generator = model.cosyvoice_model.inference_zero_shot(
                        params.text,
                        params.prompt_text,
                        prompt_speech,
                        stream=True,
                        speed=params.speed,
                        zero_shot_spk_id=params.zero_shot_spk_id
                    )
                elif params.mode == "cross_lingual":
                    inference_generator = model.cosyvoice_model.inference_cross_lingual(
                        params.text,
                        prompt_speech,
                        stream=True,
                        speed=params.speed,
                        zero_shot_spk_id=params.zero_shot_spk_id
                    )
                elif params.mode == "instruct2":
                    inference_generator = model.cosyvoice_model.inference_instruct2(
                        params.text,
                        params.instruct_text or "",
                        prompt_speech,
                        stream=True,
                        speed=params.speed,
                        zero_shot_spk_id=params.zero_shot_spk_id
                    )
                else:
                    return self.response_builder.create_error_response(
                            message,
                            ErrorCode.INVALID_PARAMS,
                            f"Unsupported mode: {params.mode}"
                        )
          
                async def process_audio():
                    try:
                        chunk_index = 0
                        for chunk in inference_generator:
                            chunk_index += 1
                            audio_data = convert_to_format(
                                chunk["tts_speech"],
                                params.output_format,
                                model.cosyvoice_model.sample_rate,
                                params
                            )

                            await websocket.send_json(
                                self.response_builder.create_response(
                                    message=message,
                                    message_type="AUDIO_RESPONSE",
                                    payload={
                                        "session_id": session_id,
                                        "text_index": current_text_index,  # 添加文本索引
                                        "chunk_index": chunk_index,
                                        "audio_data": base64.b64encode(audio_data).decode('utf-8')
                                    }
                                )
                            )
                    except Exception as e:
                        logger.error(f"Error in audio processing: {e}")
                        raise
                    finally:
                        self.audio_tasks_status[session_id]["completed_texts"] += 1
                        logger.info(f"Completed audio generation for text {current_text_index} "
                                    f"(Completed: {self.audio_tasks_status[session_id]['completed_texts']})")

                # 创建任务并存储
                task = asyncio.create_task(process_audio())
                self.pending_tasks[session_id].append(task)

                return self.response_builder.create_response(
                    message=message,
                    message_type="TEXT_PROCESSING",
                    payload={
                        "session_id": session_id,
                        "text_index": current_text_index,
                        "status": "processing"
                    }
                )

            except Exception as e:
                async with self.session_manager.lock:
                    self.audio_tasks_status[session_id]["received_texts"] -= 1
                logger.error(f"Error processing text request: {str(e)}")
                raise

        finally:
            if temp_file:
                cleanup_temp_file(temp_file)
                
    # request: SYNTHESIS_END / response: SYNTHESIS_END_ACK  
    async def handle_synthesis_end(self, websocket: WebSocket, message: dict) -> dict:
        try:
            is_valid, error = validate_message(message, self.supported_versions)
            if not is_valid:
                return self.response_builder.create_error_response(
                    message,
                    ErrorCode.INVALID_PARAMS,
                    "Invalid message",
                    error
                )
            session_id = message["payload"].get("session_id")
            if not session_id:
                return self.response_builder.create_error_response(
                    message,
                    ErrorCode.INVALID_SESSION,
                    "Invalid session"
                )

            try:
                # 等待所有音频处理任务完成
                if session_id in self.pending_tasks:
                    pending_tasks = self.pending_tasks[session_id]
                    if pending_tasks:
                        logger.info(f"Waiting for {len(pending_tasks)} pending tasks to complete")
                        await asyncio.gather(*pending_tasks)
                        
                        # 验证所有文本是否都已完成
                        status = self.audio_tasks_status.get(session_id, {})
                        received = status.get("received_texts", 0)
                        completed = status.get("completed_texts", 0)
                        
                        if completed < received:
                            logger.error(f"Not all texts completed: {completed}/{received}")
                            return self.response_builder.create_error_response(
                                message,
                                ErrorCode.SYNTHESIS_ERROR,
                                f"Not all texts completed: {completed}/{received}"
                            )
                            
                        logger.info(f"All audio processing tasks completed ({completed}/{received} texts)")
                    
                    # 清理任务列表
                    self.pending_tasks[session_id] = []
                    if session_id in self.audio_tasks_status:
                        del self.audio_tasks_status[session_id]

                return self.response_builder.create_response(
                    message=message,
                    message_type="SYNTHESIS_END_ACK",
                    payload={
                        "session_id": session_id,
                        "status": "completed"
                    }
                )

            except Exception as e:
                logger.error(f"Error waiting for tasks: {e}")
                return self.response_builder.create_error_response(
                    message,
                    ErrorCode.SYNTHESIS_ERROR,
                    "Error completing audio synthesis",
                    str(e)
                )

        except Exception as e:
            logger.error(f"Failed to handle synthesis end request: {str(e)}")
            logger.error(traceback.format_exc())
            return self.response_builder.create_error_response(
                message,
                ErrorCode.INTERNAL_ERROR,
                "Internal server error",
                str(e)
            )

            
    # request: SESSION_END / response: SESSION_END_ACK
    async def handle_session_end(self, websocket: WebSocket, message: dict) -> dict:
        try:
            is_valid, error = validate_message(message, self.supported_versions)
            if not is_valid:
                return self.response_builder.create_error_response(
                    message,
                    ErrorCode.INVALID_PARAMS,
                    "Invalid message",
                    error
                )
                
            session_id = message["payload"].get("session_id")
            if not session_id:
                return self.response_builder.create_error_response(
                    message,
                    ErrorCode.INVALID_SESSION,
                    "Invalid session"
                )

            try:
                # 清理会话相关的任务
                if session_id in self.pending_tasks:
                    logger.info(f"Cleaning up pending tasks for session {session_id}")
                    for task in self.pending_tasks[session_id]:
                        if not task.done():
                            logger.info("Canceling unfinished task")
                            task.cancel()
                            try:
                                await task  # 等待任务取消完成
                            except asyncio.CancelledError:
                                pass
                            except Exception as e:
                                logger.error(f"Error canceling task: {e}")
                    
                    del self.pending_tasks[session_id]
                    logger.info("Task cleanup completed")

                # 关闭会话
                await self.session_manager.close_session(session_id)
                
                return self.response_builder.create_response(
                    message=message,
                    message_type="SESSION_END_ACK",
                    payload={
                        "status": "success",
                        "session_id": session_id
                    }
                )
                
            except Exception as e:
                logger.error(f"Failed to close session due to: {str(e)}")
                logger.error(traceback.format_exc())
                return self.response_builder.create_error_response(
                    message,
                    ErrorCode.INVALID_SESSION,
                    "Failed to close session",
                    str(e)
                )
        
        except Exception as e:
            logger.error(f"Error handling session end: {str(e)}")
            logger.error(traceback.format_exc())
            return self.response_builder.create_error_response(
                message,
                ErrorCode.INTERNAL_ERROR,
                "Internal server error",
                str(e)
            )

    
    # request: DISCONNECT_REQUEST / response: DISCONNECT_RESPONSE   
    async def handle_disconnect_request(self, websocket: WebSocket) -> None:
        try:
            logger.info("Starting cleanup in websocket handler")
            
            # 获取需要清理的会话
            sessions_to_clean = []
            async with self.session_manager.lock:
                for session_id, session_data in self.session_manager.sessions.items():
                    if session_data["websocket"] == websocket:
                        sessions_to_clean.append(session_id)
                        logger.info(f"Found session to clean: {session_id}")

            # 对每个会话进行清理
            for session_id in sessions_to_clean:
                try:
                    logger.info(f"Cleaning session {session_id}")
                    
                    # 取消并清理pending tasks
                    if session_id in self.pending_tasks:
                        tasks = self.pending_tasks[session_id]
                        if tasks:
                            logger.info(f"Cancelling {len(tasks)} tasks for session {session_id}")
                            # 并行取消所有任务
                            cancel_tasks = [
                                asyncio.create_task(self._cancel_task(task)) 
                                for task in tasks
                            ]
                            # 设置超时时间
                            try:
                                await asyncio.wait_for(asyncio.gather(*cancel_tasks), timeout=5.0)
                            except asyncio.TimeoutError:
                                logger.warning(f"Task cancellation timed out for session {session_id}")
                            finally:
                                del self.pending_tasks[session_id]
                                logger.info(f"Removed pending tasks for session {session_id}")

                    # 清理音频任务状态
                    if session_id in self.audio_tasks_status:
                        del self.audio_tasks_status[session_id]
                        logger.info(f"Cleaned audio tasks status for session {session_id}")

                    # 关闭会话
                    await self.session_manager.close_session(session_id)
                    logger.info(f"Closed session {session_id}")

                except Exception as e:
                    logger.error(f"Error cleaning session {session_id}: {e}")
                    logger.error(traceback.format_exc())

            # 断开连接
            try:
                await self.connection_manager.disconnect(websocket)
                logger.info("Disconnected from connection manager")
            except Exception as e:
                logger.error(f"Error disconnecting websocket: {e}")

        except Exception as e:
            logger.error(f"Error in handle_disconnect_request: {e}")
            logger.error(traceback.format_exc())
        finally:
            logger.info("Cleanup process completed")

    async def _cancel_task(self, task):
        """Helper method to cancel a single task"""
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(f"Error cancelling task: {e}")

        
 

CONNECTION_TIMEOUT_MINUTES = int(os.getenv('CONNECTION_TIMEOUT_MINUTES', 30))
SESSION_TIMEOUT_MINUTES = int(os.getenv('SESSION_TIMEOUT_MINUTES', 30))              
connection_manager = ConnectionManager(max_connections=100, timeout_minutes=CONNECTION_TIMEOUT_MINUTES)
session_manager = SessionManager(max_sessions=50, timeout_minutes=SESSION_TIMEOUT_MINUTES)
tts_handler = TTSHandler(connection_manager, session_manager)

# async def cleanup_task():
#     while True:
#         try:
#             await connection_manager.cleanup_timeout_connections()
#             await session_manager.cleanup_timeout_sessions()
#         except Exception as e:
#             logger.error(f"Error in cleanup task: {e}")
#         finally:
#             await asyncio.sleep(60)  # 每分钟检查一次

# # 在应用启动时启动清理任务
# @router.on_event("startup")
# async def startup_event():
#     asyncio.create_task(cleanup_task())


@router.websocket("/ws/tts")
async def websocket_tts(websocket: WebSocket):
    cleanup_completed = False
    try:
        if not await connection_manager.connect(websocket):
            await websocket.close(code=1008)
            return

        while True:
            try:
                message = await websocket.receive_json()    
                
                await connection_manager.update_connection_activity(websocket)
                if await connection_manager.check_connection_timeout(websocket):
                    logger.info("Connection timeout")
                    break
                
                message_type = message["header"]["message_type"]
                logger.info(f"Received message type is {message_type}")
                response = None
                if message_type == "CONNECT_REQUEST":
                    response = await tts_handler.handle_connect_request(websocket, message)
                elif message_type == "SESSION_REQUEST":
                    response = await tts_handler.handle_session_request(websocket, message)
                elif message_type == "SYNTHESIS_START":
                    response = await tts_handler.handle_synthesis_start(websocket, message)
                elif message_type == "TEXT_REQUEST":
                    response = await tts_handler.handle_text_request(websocket, message)
                    continue
                elif message_type == "SYNTHESIS_END":
                    response = await tts_handler.handle_synthesis_end(websocket, message)
                elif message_type == "SESSION_END":
                    response = await tts_handler.handle_session_end(websocket, message)
                else:
                    response = tts_handler.response_builder.create_error_response(
                        message,
                        ErrorCode.INVALID_REQUEST,
                        f"Unsupported message type: {message_type}"
                    )
                
                if response:
                    await websocket.send_json(response)
                      
            except WebSocketDisconnect:
                logger.info("WebSocket disconnected by client")
                break
            except Exception as e:
                logger.error(f"Unexpected error in websocket loop: {str(e)}")
                logger.error(traceback.format_exc())
                break
            
    except Exception as e:
        logger.error(f"Critical error in websocket handler: {str(e)}")
        logger.error(traceback.format_exc())
        
    finally:
        if not cleanup_completed:  # 在没有清理完成时执行清理
            logger.info("Starting cleanup in websocket handler")
            try:
                await asyncio.shield(tts_handler.handle_disconnect_request(websocket))
                cleanup_completed = True
                logger.info("Cleanup completed successfully")
                
            except Exception as cleanup_error:
                logger.error(f"Error during cleanup: {str(cleanup_error)}")
                logger.error(traceback.format_exc())
                