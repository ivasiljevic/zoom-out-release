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
        collectgarbage()
        f = criterion:forward(output, targets)
        local df_do = criterion:backward(output, targets);
        model:backward(inputs, df_do);
        return f,gradParameters
    end
    if optimMethod == optim.asgd then
        _,_,average = optimMethod(feval, parameters, optimState)
    else
        optimMethod(feval, parameters, optimState)
    end 
    epoch = epoch + 1
end





