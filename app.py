import streamlit as st
from main import create_model, check_access, prepare_audio
import os
# Загрузка фейкового аудиофайла
fake_audio_file = st.file_uploader("Upload a fake audio file", type=["wav"])

if fake_audio_file is not None:
    # Сохранение загруженного файла
    with open('putin_deep_fake.wav', 'wb') as f:
        f.write(fake_audio_file.read())

    st.audio('putin_deep_fake.wav', format="audio/wav", start_time=0)

    # Обучение модели
    list_of_people = [
        ('putin_orig.wav', True, 'Путин оригинал'),
        ('putin_deep_fake.wav', False, 'Путин фейк')
    ]
    create_model(list_of_people)

    is_true_person = list_of_people[1]

    # Проверка доступа
    access_granted = check_access('putin_deep_fake.wav')

    if access_granted == is_true_person:
        st.success("Верный результат! Доступ разрешен." if access_granted else "Верный результат! Доступ запрещен.")
    else:
        st.error("Неверный результат! Доступ разрешен." if access_granted else "Неверный результат! Доступ запрещен.")

    # Удаление временного файла
    if os.path.exists('putin_deep_fake.wav'):
        os.remove('putin_deep_fake.wav')
    else:
        print("The file does not exist")
