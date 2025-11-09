import time

# Создаем лог-файл
log_filename = f'training_log_{time.strftime("%Y%m%d_%H%M%S")}.txt'
log_file = open(log_filename, 'w', encoding='utf-8')

def log_print(*args):
    """Вывод и в консоль и в файл"""
    message = ' '.join(str(arg) for arg in args)
    print(message)
    log_file.write(message + '\n')
    log_file.flush()  # сразу записываем в файл

# Закрываем лог-файл
def close ():
    log_file.close()
    log_print(f"Лог обучения сохранен в '{log_filename}'")    