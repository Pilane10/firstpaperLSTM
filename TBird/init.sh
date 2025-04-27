file="../output/tbird"
if [ -e $file ]
then
  echo "$file exists"
else
  mkdir -p $file
fi

cd $file

mkdir -p 'bert'
mkdir -p 'deeplog'
mkdir -p 'tbirdmodel'
mkdir -p 'loganomaly'


echo "folders done"
