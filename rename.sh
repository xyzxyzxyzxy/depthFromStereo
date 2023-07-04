
if [ $# -eq 0 ] 
then
  echo "Name of the directory is expected"
else
  a=1
  for i in $1/*.jpg; 
  do
    echo "renaming $i"
    mv -i -- "$i" $1/$(printf "%02d.jpg" "$a")
    let a=a+1
  done
fi

