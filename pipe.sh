make install

cd multiphaseJet

source cleanCase.sh

multiphaseD3Q19 -GPU 0

fieldConvert -fileType vts
