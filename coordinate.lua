--Create the two coordinate tensors.

function isint(n)
    return n==math.floor(n)
end

function coordinate_x(x,y)
    dim_x = x
    dim_y = y

    coord = torch.Tensor(dim_x,dim_y):zero()
    center = (dim_x+1)/2

    coord[{dim_x,dim_y}] = 1
    coord[{1,dim_y}] = 1
    coord[{1,1}] = -1
    coord[{dim_x,1}] = -1

    function edge_line(lin)
        div = dim_x - center
        div = math.ceil(div)

        for i=2,dim_x-1 do
            prev = coord[{i-1,lin}]

            if i<center then 
                coord[{i,lin}] = prev - coord[{1,lin}]/(div)
            end

            if i>center then 
                if math.ceil(center) == i then prev = 0 end 
                coord[{i,lin}] = prev + coord[{1,lin}]/(div) 
            end
        end
    end 

    edge_line(1)
    edge_line(dim_y)

    for j = 1, dim_x do 
        for i = 2, dim_y-1 do
            diff = coord[{j,dim_y}] - coord[{j,1}]
            coord[{j,i}] = diff/(dim_y-1) + coord[{j,i-1}]
        end
    end

    return coord
end

--Y COORDINATE TENSOR

function coordinate_y(x,y)
    dim_x = x
    dim_y = y

    coord = torch.Tensor(dim_x,dim_y):zero()
    center = (dim_y+1)/2

    coord[{dim_x,dim_y}] = -1
    coord[{1,dim_y}] = 1
    coord[{1,1}] = 1
    coord[{dim_x,1}] = -1

    function edge_line(lin)
        div = dim_y - center
        div = math.ceil(div)

        for i=2,dim_y-1 do
            prev = coord[{lin,i-1}]

        if i<center then 
            coord[{lin,i}] = prev - coord[{lin,1}]/(div)
        end

        if i>center then 
            if math.ceil(center) == i then prev = 0 end 
                coord[{lin,i}] = prev + coord[{lin,1}]/(div) 
            end
        end
    end 

    edge_line(1)
    edge_line(dim_x)

    for j = 1, dim_y do 
        for i = 2, dim_x-1 do
            diff = coord[{dim_x,j}] - coord[{1,j}]
            coord[{i,j}] = diff/(dim_x-1) + coord[{i-1,j}]
        end
    end

    return coord
end
