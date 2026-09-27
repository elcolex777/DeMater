# DeMater

Приложение для запикивания мата в аудио файлах.

На вход подается аудиозапись (или видео), на выходе аудиозапись, где слова с матом запикиваются.
Также бонусом возвращается полный распознанный текст аудио записи)

# Использование

### DeMater Web Chat

Веб-старница на которой в режиме чата можно отправить текст, голосовой или аудио, 
а в результате получить запиканную версию.
Сделано на основе функционала бота в телеграм.

Демо: https://46.16.36.127.nip.io:8002/

![example2](https://raw.githubusercontent.com/elcolex777/DeMater/refs/heads/main/example2.jpg)

### Бот Телеграм:

[t.me/DeMater_bot](https://t.me/DeMater_bot)
(к сожалению, версия с ботом в телеграм пока отключена из-за ограничений - нет нормального хостинга. В качестве альтернативы можно использовать DeMater Web Chat, он работает без ограничений)

Этот бот запикивает части аудио с матом.

Просто отправьте голосовое в чат или приложите аудиофайл.
В ответ бот выгрузит аудио файл с запиканными частями, а также распознанный текст.

**Список дополнительных команд.**

Посмотреть текущий список "матерных" слов:

```
/targetwords
```

Использовать свой список "матерных" слов:

```
/targetwords_set
список слов через пробел
```

Добавить свой список "матерных" слов к основному:

```
/targetwords_add
список слов через пробел
```

Сбросить свой список "матерных" слов:

```
/targetwords_reset
```

![example](https://raw.githubusercontent.com/elcolex777/DeMater/refs/heads/main/example.jpg)

### Приложение командной строки:

(в разработке)
python [demater.py](http://demater.py) --input_file=mater.wav --out_file=demater.wav

# Установка

Создать бота в телеграмм
переходим к боту @BotFather и отправляем команды создания своего бота
```
/newbot
DeMatTest_bot
DeMatTest_bot
Use this token to access the HTTP API:
<TOKEN>
```

Сохранить токен в переменной окружения:

```
set DEMATBOT_TOKEN=<TOKEN>
```

или глобально (нужно запустить консоль cmd от администратора)

```
setx DEMATBOT_TOKEN <TOKEN> /m

# linux
export DEMATBOT_TOKEN=<TOKEN>
export DEMATBOT_MODEL_PATH=models/vosk-model-small-ru-0.22

sudo cp -l /app/DeMater/demater.service /etc/systemd/system/demater.service
sudo systemctl daemon-reload
sudo systemctl enable demater.service
sudo systemctl start demater.service
#sudo systemctl restart demater.service
sudo systemctl status demater.service
```

Заполнить словарь по-умолчанию для заменяемых слов в файле words.txt (слово или фраза на строку)
В качестве основы можно загрузить набор слов отсюда:
<https://github.com/bars38/Russian_ban_words/blob/master/words.txt>
<https://github.com/FlacSy/BadWords/blob/master/badwords/resource/ru.bdw>
<https://disk.yandex.ru/i/6BPhVjuURt4YPA>

нужен ffmpeg для конвертирования аудио
sudo apt-get update && sudo apt-get install -y ffmpeg


Запуск приложения бота:

```
python demater_bot.py
```

Настройка nginx для веб-чата:

```
sudo bash -c 'cat > /etc/nginx/sites-available/demater.conf << '\''EOF'\''
server {
    listen 8002;
    server_name 46.16.36.127.nip.io;

    ssl_certificate /etc/letsencrypt/live/46.16.36.127.nip.io/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/46.16.36.127.nip.io/privkey.pem;

    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    client_max_body_size 50M;

    location / {
        proxy_pass http://127.0.0.1:8003;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        proxy_read_timeout 300s;
        proxy_connect_timeout 300s;
    }
}
EOF
ln -sf /etc/nginx/sites-available/demater.conf /etc/nginx/sites-enabled/ && nginx -t && systemctl reload nginx'

```

# Разработка

```
git clone https://github.com/elcolex777/DeMater.git
cd DeMater

python -m venv .venv

source .venv/bin/activate
.venv\\Scripts\\Activate.bat

pip install -r requirements.txt



#fastapi dev [main.py](http://main.py)


загрузить модели в папку models и распаковать

cd models

wget "https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip"
unzip vosk-model-small-ru-0.22.zip

wget "https://alphacephei.com/vosk/models/vosk-model-ru-0.42.zip"
unzip vosk-model-ru-0.42.zip

<https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip>
<https://alphacephei.com/vosk/models/vosk-model-ru-0.42.zip>

models\\vosk-model-small-ru-0.22
```