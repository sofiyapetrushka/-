# analyze_scalability.py
import matplotlib.pyplot as plt
import numpy as np
import glob

def analyze_scalability():
    """Анализ масштабируемости упрощенной версии"""
    
    # Чтение данных о времени выполнения
    data = []
    for filename in glob.glob('timing_*procs.txt'):
        with open(filename, 'r') as f:
            line = f.readline().split()
            if len(line) >= 2:
                procs = int(line[0])
                time_val = float(line[1])
                data.append((procs, time_val))
    
    if not data:
        print("Нет данных для анализа. Сначала запустите программу с разным числом процессов.")
        return
    
    # Сортировка по числу процессов
    data.sort(key=lambda x: x[0])
    procs = [d[0] for d in data]
    times = [d[1] for d in data]
    
    # Расчет ускорения и эффективности
    if len(times) > 0:
        time_seq = times[0]  # Время на 1 процессе
        speedup = [time_seq / t for t in times]
        efficiency = [s / p for s, p in zip(speedup, procs)]
    
    # Построение графиков
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # График времени выполнения
    ax1.plot(procs, times, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Число процессов')
    ax1.set_ylabel('Время выполнения (сек)')
    ax1.set_title('Время выполнения vs Число процессов')
    ax1.grid(True, alpha=0.3)
    
    # График ускорения
    ax2.plot(procs, speedup, 'ro-', linewidth=2, markersize=8, label='Фактическое ускорение')
    ax2.plot(procs, procs, 'k--', linewidth=1, label='Идеальное ускорение')
    ax2.set_xlabel('Число процессов')
    ax2.set_ylabel('Ускорение')
    ax2.set_title('Ускорение (Speedup)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # График эффективности
    ax3.plot(procs, efficiency, 'go-', linewidth=2, markersize=8)
    ax3.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Идеальная эффективность')
    ax3.set_xlabel('Число процессов')
    ax3.set_ylabel('Эффективность')
    ax3.set_title('Эффективность (Efficiency)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('scalability_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Вывод таблицы результатов
    print("\nТАБЛИЦА РЕЗУЛЬТАТОВ МАСШТАБИРУЕМОСТИ:")
    print("Процессы | Время (сек) | Ускорение | Эффективность")
    print("-" * 50)
    for i, (p, t) in enumerate(data):
        s = speedup[i] if i < len(speedup) else 0
        e = efficiency[i] if i < len(efficiency) else 0
        print(f"{p:8d} | {t:11.4f} | {s:9.2f} | {e:12.2f}")

if __name__ == "__main__":
    analyze_scalability()