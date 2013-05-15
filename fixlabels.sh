sed "s/^A\S*\>/1/1" test.txt > test1.txt
sed -i "s/^\S*,A\S*\>/1/1" test1.txt 
sed -i "s/^[^1]\S*\>/0/1" test1.txt 
