n=1
appendix="Meditron"
newname="Blank new name"
while IFS= read -r oldname; do
#  echo $oldname $newname $appendix
  truncatedoldname=${oldname%?????????}
  
  newname=$truncatedoldname$appendix
  
  newname=${newname#"'"}
  oldname=${oldname#"'"}
  oldname=${oldname%"'"}
  
  mv -vn "$oldname".txt "$newname".txt
  mv -vn "$oldname".wav "$newname".wav
done < "filename_differences.txt"