for N in 1024 2048; do
  for IT in 10 100 1000; do
    make all N=$N IT=$IT
    make all N=$N IT=$IT FLOAT=1
  done
done

