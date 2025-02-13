from astrbot.api.event import filter, AstrMessageEvent, MessageEventResult
from astrbot.api.star import Context, Star, register
from astrbot.api import logger
from astrbot.api.message_components import Record
from astrbot.core.utils.tencent_record_helper import tencent_silk_to_wav
import requests
from astrbot.api.all import *
import urllib.parse
import os
from pathlib import Path
import json
from astrbot.api.provider import ProviderRequest
import time

@register("astrbot_plugin_STT", "第九位魔神", "基于硅基流动的语音识别并回复", "1.0.0", "repo url")
class STTPlugin(Star):
    def __init__(self, context: Context, config: dict):
        super().__init__(context)
        self.config = config
        self.token = config.get('token', '')  # 提取token
        self.transcribed_text = ""  # 初始化为空字符串

    @event_message_type(EventMessageType.ALL)
    async def on_private_message(self, event: AstrMessageEvent):
        message_chain = event.get_messages()
        url = ""
        transcribed_text = ""

        for message in message_chain:
            if isinstance(message, Record):
                url = message.url
            else:
                return

            if url.startswith('file:///'):
                clean_path = urllib.parse.unquote(url[len('file:///'):])
                filed_path = os.path.normpath(clean_path)
                input_string = filed_path
                last_backslash_index = input_string.rfind('\\')
                if last_backslash_index != -1:
                    filed_path = input_string[:last_backslash_index]
                else:
                    filed_path = input_string

            time.sleep(2)

            directory = filed_path
            amr_files = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.amr')]

            if not amr_files:
                print("目录中没有找到任何 .amr 文件。")
            else:
                latest_file_path = max(amr_files, key=os.path.getmtime)
                file_path = os.path.normpath(latest_file_path)
                silk_path = file_path
                output_path = str(Path(__file__).parent / "output.wav")
                result = await tencent_silk_to_wav(silk_path, output_path)

                with open(output_path, 'rb') as audio_file:
                    files = {
                        'file': ('output.wav', audio_file, 'audio/wav')
                    }
                    data = {
                        'model': 'FunAudioLLM/SenseVoiceSmall'
                    }
                    headers = {
                        "Authorization": f"Bearer {self.token}"
                    }
                    response = requests.post("https://api.siliconflow.cn/v1/audio/transcriptions", files=files, data=data, headers=headers)
                    response.raise_for_status()
                    logger.info(f"Transcription response: {response.text}")
                    response_json = json.loads(response.text)
                    transcribed_text = response_json.get('text', '')

                    logger.info(f"Transcribed text: {transcribed_text}")
                    print(transcribed_text)
                    self.transcribed_text = transcribed_text  # 更新实例属性

                    #发送给LLM
                    func_tools_mgr = self.context.get_llm_tool_manager()
                    # 获取用户当前与 LLM 的对话以获得上下文信息。
                    curr_cid = await self.context.conversation_manager.get_curr_conversation_id(
                        event.unified_msg_origin)  # 当前用户所处对话的对话id，是一个 uuid。
                    conversation = None  # 对话对象
                    context = []  # 上下文列表
                    if curr_cid:
                        conversation = await self.context.conversation_manager.get_conversation(
                            event.unified_msg_origin, curr_cid)
                        context = json.loads(conversation.history)
                    yield event.request_llm(
                        prompt=self.transcribed_text,
                        func_tool_manager=func_tools_mgr,
                        session_id=curr_cid,  # 对话id。如果指定了对话id，将会记录对话到数据库
                        contexts=context,  # 列表。如果不为空，将会使用此上下文与 LLM 对话。
                        system_prompt="",
                    )




