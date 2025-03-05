function torchrun(cmd)
    cmd = "cd ../Training && $cmd || (sleep 10 && $cmd) || (sleep 60 && $cmd)"
    run(`sh -c $cmd`)
end

function finetune()
   run(`julia import_data.jl`)
   run(`julia transformer.jl`)
    for m in [0, 1]
        torchrun("torchrun --standalone --nproc_per_node=1 transformer.py --datadir ../../data/finetune --finetune ../../data/finetune/transformer.pt --finetune_medium $m")
    end
   run(`julia ../Training/bagofwords.jl --finetune`)
   for m in [0, 1]
       for metric in ["rating", "watch", "plantowatch", "drop"]
           torchrun("torchrun --standalone --nproc_per_node=1 bagofwords.py --datadir ../../data/finetune --medium $m --metric $metric --finetune ../../data/finetune/bagofwords.$m.$metric.pt")
       end
   end
  run(`python register.py`)
  run(`julia regress.jl`)
  cmd = "cd ../Package/Embed && julia package.jl"
  run(`sh -c $cmd`)
end

finetune()
