# API genelinde kullanilan sabit degerler burada toplanir.
# Boylece limit degistirmek icin kod icinde arama yapmaya gerek kalmaz.

# Yuklenebilecek maksimum dosya boyutu (megabyte cinsinden).
MAX_FILE_SIZE_MB = 10

# Yukaridaki degeri byte'a cevirip endpoint'te dogrudan kiyaslamada kullaniyoruz.
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# Sadece bu content-type'lara izin veriyoruz; baska bir formatta dosya gelirse
# (ornegin PDF ya da text/plain) endpoint 415 donecek.
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
