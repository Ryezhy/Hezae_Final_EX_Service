from http.client import HTTPException
from urllib.request import Request

from fastapi import FastAPI, Depends
from minio import Minio, S3Error

from starlette.responses import JSONResponse, StreamingResponse
from starlette.staticfiles import StaticFiles
import pynvml  #导包
import psutil
import torch

app = FastAPI()
# 创建一个 StaticFiles 实例，指定静态文件目录
static_files = StaticFiles(directory="static")
app.mount("/static", static_files)


# MinIO 客户端配置
minio_client = Minio(
    "127.0.0.1:9015",
    access_key="admin",
    secret_key="admin841755",
    secure=False  # 如果使用 HTTPS，则设置为 True
)

# 要查询的存储桶名称
bucket_name = "main"


@app.get("/files/")
async def list_files():
    try:
        # 获取存储桶中的对象列表
        objects = minio_client.list_objects(bucket_name)

        # 格式化文件列表
        file_list = [obj.object_name for obj in objects]

        return {"files": file_list}
    except S3Error as e:
        raise HTTPException()


@app.get("/download/{file_name}")
async def download_file(file_name: str):
    try:
        # 获取文件对象和元数据
        file_obj = minio_client.get_object(bucket_name, file_name)
        file_stat = minio_client.stat_object(bucket_name, file_name)
        file_size = file_stat.size

        # 构造头部,包含文件名和文件大小
        headers = {
            'Content-Disposition': f'attachment; filename={file_name}; size={file_size}',
            "Content-Length": f"{file_size}"
        }

        # 返回 StreamingResponse
        return StreamingResponse(
            file_obj.stream(),
            media_type="application/octet-stream",
            headers=headers
        )

    except S3Error as e:
        if e.code == "NoSuchKey":
            return JSONResponse({"message": "文件不存在"}, status_code=404)
        else:
            raise HTTPException()


@app.middleware("http")
async def allow_origin(request: Request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response


@app.get("/")
async def get():
    #在静态文件中读取json
    return JSONResponse({"message": "Hello World"})


#检查支不支持cuda
@app.get("/check_cuda")
async def check_cuda():
    try:
        import torch
        if torch.cuda.is_available():
            print("CUDA is available")
            gpu_name = torch.cuda.get_device_name(0)
            print("GPU name:", gpu_name)
            cuda_versions = torch.version.cuda
            print("CUDA version:", cuda_versions)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            gpu_memory = round(gpu_memory, 2)
            print("Memory:", gpu_memory, 'GB')
            return JSONResponse({"message": "CUDA is available", "gpu_name": gpu_name, "cuda_version": cuda_versions,
                                 "gpu_memory": f"{gpu_memory}GB"})
        else:
            print("CUDA is not available")
            return JSONResponse({"message": "CUDA is not available"})
    except ImportError:
        print("PyTorch is not installed")
        return JSONResponse({"message": "PyTorch is not installed"})


def get_cpu_info():
    """获取 CPU 信息并返回数组格式"""
    cpu_count = psutil.cpu_count(logical=False)  # 物理核心数
    cpu_count_logical = psutil.cpu_count(logical=True)  # 逻辑核心数
    cpu_percent = psutil.cpu_percent(interval=1)  # 获取过去 1 秒的 CPU 使用率
    cpu_freq = psutil.cpu_freq()
    cpu_percents = psutil.cpu_percent(percpu=True, interval=1)  # 每个 CPU 核心的使用率

    return [
        {"name": "核心数", "value": cpu_count},
        {"name": "逻辑核心数", "value": cpu_count_logical},
        {"name": "基准频率", "value": f"{cpu_freq.current / 1000:.2f} GHz"},
        {"name": "使用率", "value": f"{cpu_percent}%"},
        {"name": "每个核心使用率", "value": [f" CPU {i + 1}: {percent}%" for i, percent in enumerate(cpu_percents)]}
    ]


def get_memory_info():
    """获取内存信息并返回数组格式"""
    memory = psutil.virtual_memory()
    memory_total = memory.total / 1024 / 1024 / 1024  # GB
    memory_used = memory.used / 1024 / 1024 / 1024  # GB
    memory_free = memory.free / 1024 / 1024 / 1024  # GB
    memory_percent = memory.percent
    memory_swap = psutil.swap_memory()
    memory_swap_total = memory_swap.total / 1024 / 1024 / 1024  # GB
    memory_swap_used = memory_swap.used / 1024 / 1024 / 1024  # GB
    memory_swap_free = memory_swap.free / 1024 / 1024 / 1024  # GB

    return [
        {"name": "总容量", "value": f"{memory_total:.2f} GB"},
        {"name": "已使用", "value": f"{memory_used:.2f} GB"},
        {"name": "剩余", "value": f"{memory_free:.2f} GB"},
        {"name": "使用率", "value": f"{memory_percent}%"},
        {"name": "交换分区总容量", "value": f"{memory_swap_total:.2f} GB"},
        {"name": "交换分区已使用", "value": f"{memory_swap_used:.2f} GB"},
        {"name": "交换分区剩余", "value": f"{memory_swap_free:.2f} GB"},
    ]


def get_gpu_info():
    """获取 GPU 信息并返回数组格式"""
    pynvml.nvmlInit()
    gpu_count = pynvml.nvmlDeviceGetCount()
    UNIT = 1024 * 1024
    gpu_details = []
    for i in range(gpu_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        memoryInfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpuTemperature = pynvml.nvmlDeviceGetTemperature(handle, 0)
        gpuUtilRate = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        gpuMemoryRate = pynvml.nvmlDeviceGetUtilizationRates(handle).memory
        gpu_name = torch.cuda.get_device_name(i)

        # 将 GPU 信息存入一个字典
        gpu_data = {
            "GPU ID": i,
            "GPU 名称": gpu_name,
            "内存总容量": f"{memoryInfo.total / UNIT:.2f} MB",
            "使用容量": f"{memoryInfo.used / UNIT:.2f} MB",
            "剩余容量": f"{memoryInfo.free / UNIT:.2f} MB",
            "显存空闲率": f"{memoryInfo.free / memoryInfo.total:.2%}",  # 添加百分号
            "温度": f"{gpuTemperature} °C",
            "GPU 计算核心使用率": f"{gpuUtilRate}%",  # 添加百分号
            "GPU 内存使用率": f"{gpuMemoryRate}%",  # 添加百分号
            "内存占用率": f"{memoryInfo.used / memoryInfo.total:.2%}"  # 添加百分号
        }

        # 将字典转换为 {name: 'GPUID', value: 0} 格式的数组
        for name, value in gpu_data.items():
            gpu_details.append({"name": name, "value": value})

    # 返回数组
    return gpu_details


@app.get("/get_hardware_info")
async def get_hardware_info(hardware: str):
    """获取硬件信息，通过 get 参数 `hardware` 指定硬件类型"""
    try:
        if hardware == "cpu":
            return JSONResponse(get_cpu_info())
        elif hardware == "memory":
            return JSONResponse(get_memory_info())
        elif hardware == "gpu":
            return JSONResponse(get_gpu_info())
        else:
            return JSONResponse({"message": "无效的硬件类型"}, status_code=400)

    except ImportError:
        if hardware == "cpu" or hardware == "memory":
            print("psutil 未安装")
            return JSONResponse({"message": "psutil 未安装"})
        elif hardware == "gpu":
            print("pynvml 或 torch 未安装")
            return JSONResponse({"message": "pynvml 或 torch 未安装"})
