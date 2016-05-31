--Zoomout model training.

function train(model,inputs, targets)
    epoch = epoch or 1
-- do one epoch
    print("==> online epoch # " .. epoch ..']')
-- create closure to evaluate f(X) and df/dX
        local feval = function(x)
        if x ~= parameters then
        parameters:copy(x)
        end
        gradParameters:zero()
        local f = 0
        local output = model:forward(inputs);
        f = criterion:forward(output, targets)
        local df_do = criterion:backward(output, targets);
        model:backward(inputs, df_do);

    return f,gradParameters
end
-- optimize on current mini-batch
    if optimMethod == optim.asgd then
        _,_,average = optimMethod(feval, parameters, optimState)
    else
        optimMethod(feval, parameters, optimState)
    end
   
    local filename = paths.concat('results', 'model.net')
    os.execute('mkdir -p ' .. sys.dirname(filename))
    if epoch% 1000 == 0 then 
    torch.save(filename, model) 
    end
    epoch = epoch + 1
end





