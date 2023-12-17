# AS project barebones

## Installation guide

```shell
pip install -r ./requirements.txt
```

## Training guide

```shell
python3 train.py -c hw_nv/configs/train.json
```
В train.py стоит 100 эпох, и, скорее всего, если обучить прям 100, то все будет вообще прекрасно, но у меня столько времени не было(
Поэтому я останавливал примерно через на 8-10к шагов

## Testing guide

```shell
python3 test.py -r <path_to_checkpoint> -t hw_as/test_data
```
По умолчанию подразумевается, что вместе с чекпоинтом лежит и конфиг. Чекпоинты моей модели лежат на гугл диске: [чекпоинт](https://drive.google.com/file/d/1q5ctqoJk7KzTnVdtEUKVqlGbGoKu5wau/view?usp=sharing), [конфиг](https://drive.google.com/file/d/1S4E4aphovX3Oq3168_VmUTfK7YAe260I/view?usp=sharing). Папку датасета можно менять, но я загрузил туда все необходимые файлы - 3 файла из датасета в кагле, 3 файла, сгенерированные моей моделью из 4 ДЗ, а также 2 файла из интернета, один с bonafide аудио, другой с spoof (они подписаны)
При запуске данной команды на стандартный вывод будет выведена информация о файлах, название (точнее, путь до файла), полученные логиты и вероятность того, что это bonafide аудио.
