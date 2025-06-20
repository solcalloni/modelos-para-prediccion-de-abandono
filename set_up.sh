#/bin/bash
read -p "Estás seguro de que querés crear el entorno virtual? (yes/no) " yn

case $yn in 
	yes ) echo OK, ahí vamos;;
	no ) echo Abort!;
		exit;;
	* ) echo Respuesta incorrecta;
		exit 1;;
esac
#

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt