./sbin/rvkde --best --cv --regress --mae -n 5 -v result/val/train_8_10.scale -b 1,5,.5 --ks 1,34,2 --kt 1,100,2 > result/val/rvkde_8_10-train-result
./sbin/rvkde --best --cv --regress --mae -n 5 -v result/val/train_17_19.scale -b 1,5,.5 --ks 1,34,2 --kt 1,100,2 > result/val/rvkde_17_19-train-result
# svm-grid -log2c -8,8,1 -log2g -8,8,1 -v 5 train_8_10.scale > train_arg_8_10
# svm-grid -log2c -8,8,1 -log2g -8,8,1 -v 5 train_17_19.scale > train_arg_17_19
