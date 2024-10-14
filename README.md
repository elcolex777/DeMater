# DeMater
Приложение для запикивания мата в аудио файлах.

На вход подается аудиозапись, на выходе аудиозапись, где слова с матом запикиваются.
Также бонусом возвращается полный распознанный текст аудио записи)

# Использование

### Бот Телеграм:
[t.me/DeMater_bot](https://t.me/DeMater_bot)

Этот бот запикивает части аудио с матом.

Просто отправьте голосовое в чат или приложите аудиофайл.
В ответ бот выгрузит аудио файл с запиканными частями, а также распознанный текст.


**Список дополнительных команд.**

Посмотреть текущий список "матерных" слов:

/targetwords

Использовать свой список "матерных" слов

/targetwords список,слов,через,запятую

Сбросить свой список "матерных" слов

/targetwords_reset


### Приложение командной строки:
(в разработке)
python demater.py --input_file=mater.wav --out_file=demater.wav

# Установка

Создать бота в телеграмм
переходим к боту @BotFather и отправляем команды создания своего бота
/newbot
DeMatTest_bot
DeMatTest_bot
Use this token to access the HTTP API:
<TOKEN>

Сохранить токен в переменной окружения:
set DEMATBOT_TOKEN=<TOKEN>

или глобально (нужно запустить консоль cmd от администратора)
setx DEMATBOT_TOKEN <TOKEN> /m

Заполнить словарь по-умолчанию для заменяемых слов в файле words.txt (слово или фраза на строку)
В качестве основы можно загрузить набор слов отсюда:
https://github.com/bars38/Russian_ban_words/blob/master/words.txt

Запуск приложения бота:
python demater_bot.py

# Разработка

python -m venv .venv
.venv\Scripts\Activate.bat

pip install -r requirements.txt

fastapi dev main.py

загрузить модели в папку models и распаковать
https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip
https://alphacephei.com/vosk/models/vosk-model-ru-0.42.zip

models\vosk-model-small-ru-0.22