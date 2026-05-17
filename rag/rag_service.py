# 导入必要的库和模块
from langchain_core.documents import Document  # 文档对象，用于存储文本内容和元数据
from langchain_core.output_parsers import StrOutputParser  # 字符串输出解析器，将模型输出转换为字符串
from rag.vector_store import VectorStoreService  # 向量存储服务，用于存储和检索文档
from utils.prompt_loader import load_rag_prompts  # 提示词加载器，读取预设的提示模板
from langchain_core.prompts import PromptTemplate  # 提示模板类，用于构建给AI的提示信息
from model.factory import chat_model  # 聊天模型实例，实际的AI模型


# 打印提示词的辅助函数，用于调试查看最终发送给模型的提示内容
def print_prompt(prompt):
    """
    打印并返回提示词对象
    :param prompt: 提示词对象
    :return: 原样返回提示词对象（用于链式调用中不中断流程）
    """
    print("=" * 20)  # 打印分隔线，便于在控制台区分
    print(prompt.to_string())  # 将提示词转换为字符串并打印
    print("=" * 20)  # 打印分隔线
    return prompt  # 返回提示词，保证数据流继续传递


class RagSummarizeService(object):
    """
    RAG总结服务类
    工作流程：用户提问 -> 搜索相关参考资料 -> 将问题和资料一起交给AI模型 -> AI生成总结回答
    """

    def __init__(self):
        """
        初始化方法：创建对象时自动执行，准备所有需要的组件
        """
        # 创建向量存储服务对象，用于管理和检索文档
        self.vector_store = VectorStoreService()

        # 从向量存储中获取检索器，用于根据问题查找相关文档
        self.retriever = self.vector_store.get_retriever()

        # 加载RAG提示词模板文本
        self.prompt_text = load_rag_prompts()

        # 将提示词文本转换为PromptTemplate对象，方便后续使用
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)

        # 获取聊天模型实例（如GPT、文心一言等AI模型）
        self.model = chat_model

        # 初始化处理链，将各个组件串联起来形成完整的处理流程
        self.chain = self._init_chain()

    def _init_chain(self):
        """
        初始化处理链
        处理链就像一个流水线，数据从左到右依次经过每个环节
        流程：提示模板 -> 打印提示词(调试用) -> AI模型处理 -> 解析输出结果
        """
        # 使用 | 符号连接各个组件，形成处理管道
        # 数据流向：prompt_template -> print_prompt -> model -> StrOutputParser
        chain = self.prompt_template | print_prompt | self.model | StrOutputParser()
        return chain  # 返回构建好的处理链

    def retriever_docs(self, query: str) -> list[Document]:
        """
        根据用户问题检索相关文档
        :param query: 用户的问题，例如"小户型适合哪些扫地机器人"
        :return: 相关文档列表，每个文档包含文本内容和元数据
        """
        # 调用检索器的invoke方法，传入问题，返回匹配的文档
        return self.retriever.invoke(query)

    def rag_summarize(self, query: str) -> str:
        """
        RAG总结主方法：接收问题，检索资料，生成总结回答
        :param query: 用户的问题
        :return: AI生成的总结回答字符串
        """
        # 第一步：根据问题检索相关的参考文档
        context_docs = self.retriever_docs(query)

        # 第二步：将所有参考文档整理成一个字符串
        context = ""  # 初始化空字符串，用于拼接所有参考内容
        counter = 0  # 计数器，用于标记第几个参考资料

        # 遍历每个检索到的文档
        for doc in context_docs:
            counter += 1  # 计数器加1
            # 将每个文档的内容和元数据格式化后拼接到context中
            # doc.page_content: 文档的文本内容
            # doc.metadata: 文档的元数据（如来源、页码等信息）
            context += f"【参考资料{counter}】: 参考资料：{doc.page_content} | 参考元数据：{doc.metadata}\n"

        # 第三步：调用处理链，将问题和参考资料一起交给AI模型生成回答
        return self.chain.invoke(
            {
                "input": query,  # 用户的问题
                "context": context,  # 整理好的参考资料
            }
        )


# 程序入口：当直接运行此文件时执行
if __name__ == '__main__':
    # 创建RAG总结服务对象
    rag = RagSummarizeService()

    # 测试：询问"小户型适合哪些扫地机器人"，并打印回答结果
    print(rag.rag_summarize("小户型适合哪些扫地机器人"))