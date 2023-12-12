import streamlit as st
from main import create_model, check_access, prepare_audio
import os

list_of_people = [
    ('putin_orig.wav', True, 'Путин оригинал'),
    ('putin_deep_fake.wav', False, 'Путин фейк')
]

st.title("Проверка доступа по аудио")

fake_audio_file = st.file_uploader("Загрузите аудиофайл", type=["wav"])

if fake_audio_file is not None:
    temp_file_path = "temp_audio.wav"

    st.audio(temp_file_path, format="audio/wav", start_time=0)

    with open(temp_file_path, 'wb') as f:
        f.write(fake_audio_file.read())

    create_model(list_of_people)

    access_granted = check_access(temp_file_path)

    if access_granted:
        st.success("Доступ разрешен!")
    else:
        st.error("Доступ запрещен!")

    if os.path.exists(temp_file_path):
        os.remove(temp_file_path)
    else:
        st.warning("Не удалось удалить временный файл")
