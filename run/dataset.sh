echo mkdir -p data
mkdir -p data
echo Downloading...
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1e-OFd6kMtz1x1zO0RQh5hVYfD_yfdJii' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1e-OFd6kMtz1x1zO0RQh5hVYfD_yfdJii" -O data/replica.zip && rm -rf /tmp/cookies.txt
echo unzip data/replica.zip -d data/
unzip data/replica.zip -d data/
echo rm data/replica.zip
rm data/replica.zip

echo wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1nt1F-GxDwkQXOIyO-VZwy230fO7iqLam' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1nt1F-GxDwkQXOIyO-VZwy230fO7iqLam" -O data/TanksAndTemple.zip && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1nt1F-GxDwkQXOIyO-VZwy230fO7iqLam' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1nt1F-GxDwkQXOIyO-VZwy230fO7iqLam" -O data/TanksAndTemple.zip && rm -rf /tmp/cookies.txt
echo unzip data/TanksAndTemple.zip -d data/
unzip data/TanksAndTemple.zip -d data/
echo mv data/data/TanksAndTemple data/
mv data/data/TanksAndTemple data/
echo rm -r data/data
rm -r data/data
echo rm data/TanksAndTemple.zip
rm data/TanksAndTemple.zip 