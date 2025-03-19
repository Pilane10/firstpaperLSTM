file="../output/bgl"
if [ -e "$file" ]; then
  echo "$file exists"
else
  mkdir -p "$file"
fi

# Now create the bert folder inside bgl
bert_dir="../output/bgl/bert"
if [ -e "$bert_dir" ]; then
  echo "$bert_dir exists"
else
  mkdir -p "$bert_dir"
fi
