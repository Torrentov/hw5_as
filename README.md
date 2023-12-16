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
python3 test.py -r <path_to_checkpoint>
```
По умолчанию подразумевается, что вместе с чекпоинтом лежит и конфиг. Чекпоинты моей модели лежат на [гугл диске](https://drive.google.com/drive/folders/1Q4Xp7BrSjqDY5LwqdGgMMZaBVc_ppBqc?usp=drive_link). Не всякий случай, вместе с model_best оставил еще и чекпоинт эпохи, но лучше брать model_best, конечно. При запуске этого кода автоматически в папке test_results/results/ появятся файлы 0.wav, 1.wav и 2.wav, соответствующие 3 тестовым аудио из чата

