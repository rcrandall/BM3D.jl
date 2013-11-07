require("bm3d")

using Images, ImageView
using NumericExtensions
using BM3D

function getPsnr(img::Matrix{Float64},imgClean::Matrix{Float64})
	return 20.0*log10(255.0*sqrt(length(imgClean)/sumsqdiff(imgClean,img)))
end

# Convert test image to grayscale
img = imread("lena.tiff")
tmp = mean(convert(Array{Float64},img.data),1)
tmp = reshape(tmp,512,512)'
img = convert(Image,tmp)
img.properties["limits"] = (0.0,255.0)

# Noisy image
imgn = deepcopy(img)
sig = 15.0
imgn.data += sig*randn(512,512)

# Perform BM3D denoising
@time imgBasic = bm3d_thr(imgn.data,sig)
@time imgOut = bm3d_wie(imgn.data,imgBasic,sig)

imgBasic = convert(Image,imgBasic)
imgBasic.properties["limits"] = (0.0,255.0)

imgOut = convert(Image,imgOut)
imgOut.properties["limits"] = (0.0,255.0)

errNoisy = getPsnr(imgn.data,img.data)
errBasic = getPsnr(img.data,imgBasic.data)
errFinal = getPsnr(img.data,imgOut.data)

println("PSNR noisy: ", errNoisy, " dB")
println("PSNR basic: ", errBasic, " dB")
println("PSNR denoised: ", errFinal, " dB")

#display([imgn imgBasic])