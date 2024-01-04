std::string get_image_type(const cv::Mat& img, bool more_info=true) 
{
    std::string r;
    int type = img.type();
    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);

    switch (depth) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
    }

    r += "C";
    r += (chans + '0');
   
    if (more_info)
        std::cout << "depth: " << img.depth() << " channels: " << img.channels() << std::endl;

    return r;
}

void show_image(cv::Mat& img, std::string title)
{
    std::string image_type = get_image_type(img);
    cv::namedWindow(title + " type:" + image_type, cv::WINDOW_NORMAL); // Create a window for display.
    cv::imshow(title, img);
    cv::waitKey(0);
}

auto transpose(at::Tensor tensor, c10::IntArrayRef dims = { 0, 3, 1, 2 })
{
    std::cout << "############### transpose ############" << std::endl;
    std::cout << "shape before : " << tensor.sizes() << std::endl;
    tensor = tensor.permute(dims);
    std::cout << "shape after : " << tensor.sizes() << std::endl;
    std::cout << "######################################" << std::endl;
    return tensor;
}

auto ToTensor(cv::Mat img, bool show_output = false, bool unsqueeze=false, int unsqueeze_dim = 0)
{
    std::cout << "image shape: " << img.size() << std::endl;
    at::Tensor tensor_image = torch::from_blob(img.data, { img.rows, img.cols, 3 }, at::kByte);

    if (unsqueeze)
    {
        tensor_image.unsqueeze_(unsqueeze_dim);
        std::cout << "tensors new shape: " << tensor_image.sizes() << std::endl;
    }
    
    if (show_output)
    {
        std::cout << tensor_image.slice(2, 0, 1) << std::endl;
    }
    std::cout << "tenor shape: " << tensor_image.sizes() << std::endl;
    return tensor_image;
}

auto ToInput(at::Tensor tensor_image)
{
    // Create a vector of inputs.
    return std::vector<torch::jit::IValue>{tensor_image};
}

auto ToCvImage(at::Tensor tensor)
{
    int width = tensor.sizes()[0];
    int height = tensor.sizes()[1];
    try
    {
        cv::Mat output_mat(cv::Size{ height, width }, CV_8UC3, tensor.data_ptr<uchar>());
        
        show_image(output_mat, "converted image from tensor");
        return output_mat.clone();
    }
    catch (const c10::Error& e)
    {
        std::cout << "an error has occured : " << e.msg() << std::endl;
    }
    return cv::Mat(height, width, CV_8UC3);
}

int main(int argc, const char* argv[]) 
{
    std::string msg = "sample image";
    auto currentPath = "cpp\\port\\LibtorchPort\\imgs\\img1.jpg";
    auto img = cv::imread(currentPath);
    show_image(img, msg);

    // convert the cvimage into tensor
    auto tensor = ToTensor(img);

    // preprocess the image. meaning alter it in a way a bit!
    tensor = tensor.clamp_max(c10::Scalar(50));

    auto cv_img = ToCvImage(tensor);
    show_image(cv_img, "converted image from tensor");
    // convert the tensor into float and scale it 
    tensor = tensor.toType(c10::kFloat).div(255);
    // swap axis 
    tensor = transpose(tensor, { (2),(0),(1) });
    //add batch dim (an inplace operation just like in pytorch)
    tensor.unsqueeze_(0);

    auto input_to_net = ToInput(tensor);
   

    torch::jit::script::Module r18;

    try 
    {
        std::string r18_model_path = "D:\\Codes\\python\\Model_Zoo\\jitcache\\resnet18.pt";


        // Deserialize the ScriptModule from a file using torch::jit::load().
        r18 = torch::jit::load(r18_model_path);
    
        // Execute the model and turn its output into a tensor.
        at::Tensor output = r18.forward(input_to_net).toTensor();

        //sizes() gives shape. 
        std::cout << output.sizes() << std::endl;
        std::cout << "output: " << output[0] << std::endl;
        //std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';
    
    }
    catch (const c10::Error& e) 
    {
        std::cerr << "error loading the model\n" <<e.msg();
        return -1;
    }

    std::cout << "ok\n";
    std::system("pause");
    return 0;
}