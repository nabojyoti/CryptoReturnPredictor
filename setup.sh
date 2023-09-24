# GREEN='\033[0;32m'

 

# echo [$(date)]: "START"

 

# echo [$(date)]: "=========================="

# echo [$(date)]: "Creating a new Environment"

# conda create --prefix ./eth_env python==3.9 -y

 

# echo [$(date)]: "Environment created with name 'eth_env'"

# echo [$(date)]: "Activate the conda eth_env"

 

 

# echo [$(date)]: "---------- Environment Live ----------"

# echo [$(date)]: ""

# echo [$(date)]: "Installing required Libraries. Please wait ..."

# echo [$(date)]: ""

# pip install -r requirements.txt

 

# echo -e "${GREEN}\nLibraries Installed Successfully."

# echo [$(date)] "================================="

 

# echo [$(date)]: "END"


echo [$(date)]: 'START'
echo [$(date)]: 'Creating conda env with python 3.9'
conda create --prefix ./eth_env python=3.9 -y
echo [$(date)]: 'activate env'
source activate ./eth_env
echo [$(date)]: 'installing TA-LIB'
conda install -c conda-forge ta-lib -y
echo [$(date)]: 'installing requirements'
pip3 install -r requirements.txt
echo [$(date)]: 'Setup END'

#RUN : bash setup.sh