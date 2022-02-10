import psutil
import GPUtil


def get_stats():
    stats = {}
    gpu = GPUtil.getGPUs()[0]
    stats["gpu"] = int(gpu.load*100)
    stats["gpu_ram"] = int(gpu.memoryUtil*100)
    stats["cpu"] = int(psutil.cpu_percent())
    stats["cpu_ram"] = int(psutil.virtual_memory().percent)
    return stats

if __name__ == "__main__":
    stats = get_stats()
    print(stats)
