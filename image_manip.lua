require 'torch'
require 'cutorch'
require 'cunn'
require 'image'
require 'qt'


function get_augmentations(num_augmentations,pct_augmentations,input_batch,mask_batch)
    n = input_batch:size()[1]		
    input_out = torch.Tensor(n + n * num_augmentations,input_batch:size()[2],input_batch:size()[3],input_batch:size()[4]):float()
    mask_out = torch.Tensor(n + n * num_augmentations,input_batch:size()[2],input_batch:size()[3],input_batch:size()[4]):float()

    input_out[{{1,n},{},{},{}}] = input_batch:clone()
    mask_out[{{1,n},{},{},{}}] = mask_batch:clone()

    for aug = 0,num_augmentations-1 do

        --decide which augmentations to apply
        --choices = torch.zeros(5)	
        choices = torch.zeros(3)	
        while choices:sum() == 0 do
            --choices = torch.rand(5)
            choices = torch.rand(3)
            choices[choices:lt(1 - pct_augmentations)] = 0
            choices[choices:gt(1 - pct_augmentations)] = 1
        end

       --decide the order to apply them
       --order = torch.randperm(5)
       order = torch.randperm(3)


        for j = 1,n do
            for k = 1,3 do
                if order[k] == 1 and choices[1] == 1 then
	            a = input_out[j]
                    b = mask_out[j]
                    input_out[n + aug*n + j],mask_out[n + aug*n + j] = rotate_image(input_out[j],mask_out[j])
                elseif order[k] == 2 and choices[2] == 1 then
                    a = input_out[j]
                    b = mask_out[j]
                    input_out[n + aug*n + j],mask_out[n + aug*n + j] = scale_and_stretch(a,b)
                elseif order[k] == 3 and choices[3] == 1 then
                    input_out[n + aug*n + j],mask_out[n + aug*n + j] = change_contrast(input_out[j],mask_out[j])
                end
            end
        end
    end

    return input_out,mask_out
end
			
			 
function rotate_image(im,mask)
    --get the random rotation - lets say += 5 - 40 degrees
	
    rotation = torch.rand(1)[1] * (0.61087) - (0.61087 / 2)
    if rotation > 0 then
        rotation = rotation + 0.08727
    else 
        rotation = rotation -0.08727
    end

    new_im = image.rotate(im,rotation)
	
    if mask:sum() > 0 then
        new_mask = image.rotate(mask,rotation)
    end
	
    return new_im,new_mask
end

--call this after scaling and stretching - or anything else that changes the size
function fix_image(new_image,original_cols,original_rows)
    new_rows = new_image:size()[new_image:dim()-1]
    new_cols = new_image:size()[new_image:dim()]

    --left pad if we need to	
    if  new_cols < original_cols then
        if new_image:dim() == 2 then
            new_image = torch.zeros(new_rows,original_cols - new_cols):float():cat(new_image,2)
        elseif new_image:dim() == 3 then
            new_image = torch.zeros(new_image:size()[1],new_rows,original_cols - new_cols):float():cat(new_image,3)
        end
    end
	
    --bottom pad if we need to
    if  new_rows < original_rows then
        if new_image:dim() == 2 then
            bottom_pad = torch.zeros(original_rows - new_rows,math.max(new_cols,original_cols)):float()
            new_image = torch.cat(new_image,bottom_pad,1)
        elseif new_image:dim() == 3 then
            bottom_pad = torch.zeros(new_image:size()[1],original_rows - new_rows,math.max(new_cols,original_cols)):float()
            new_image = torch.cat(new_image,bottom_pad,2)
        end
    end
	
    --crop from top right
    return image.crop(new_image,'tr',original_cols,original_rows)
end


function scale_and_stretch(im,mask)
    --get a random scaling on both axes which will lead to some "warping"
    --say .8 to 1.2

    original_rows = im:size()[im:dim()-1]
    original_cols = im:size()[im:dim()]
	
    row_factor = torch.rand(1)[1] * .4 + .8
    col_factor = torch.rand(1)[1] * .4 + .8


    row_size = math.floor(original_rows * row_factor)
    col_size = math.floor(original_cols * col_factor)

    new_im = image.scale(im,col_size, row_size,'bilinear')
    new_im = fix_image(new_im,original_cols,original_rows)

    if mask:sum() > 0 then
        new_mask = image.scale(mask,col_size, row_size,'bilinear')
        new_mask = fix_image(new_mask,original_cols,original_rows)
    end

    return new_im,new_mask
end

--this is simplistic - I mean-centered the images - this will increase the scale
function change_contrast(im,mask)
	factor = torch.rand(1)[1]  + .5
	new_im = im:mul(factor)
	new_mask = mask:mul(factor)

	return new_im,new_mask
end

function add_noise(im,mask)
	--so the noise will be gaussian - mean zero, std 1
	--dividing this by a large number keeps this reasonable
	r = im:size()[im:dim()-1]
	c = im:size()[im:dim()]	

	factor = torch.rand(1)[1] * 20 + 5
	noise = torch.randn(r,c):mul(1/factor):float()

	--decide where to apply the noise
	factor = torch.rand(1)[1]
	noise_mask = torch.rand(r,c):float()
	noise_mask[noise_mask:gt(factor)] = 1
	noise_mask[noise_mask:lt(1)] = 0
	noise = noise:cmul(noise_mask)

	if im:dim() == 2 then
		im = im:add(noise)
	elseif im:dim() == 3 then
		for i = 1,im:size()[1] do
			im[i] = im[i]:add(noise)
		end
	end

	return im,mask
end

function blur(im,mask)
	original_rows = im:size()[im:dim()-1]
	original_cols = im:size()[im:dim()]

	--get a gaussian kernel
	factor = math.ceil(torch.rand(1)[1] * 22) + 1
	kernel = image.gaussian({size=factor})
	im = image.convolve(im,kernel)

	im = fix_image(im,original_cols,original_rows)

	return im,mask
end





