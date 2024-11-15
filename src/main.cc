// dear imgui: standalone example application for GLFW + OpenGL 3, using programmable pipeline
// If you are new to dear imgui, see examples/README.txt and documentation at the top of imgui.cpp.
// (GLFW is a cross-platform general purpose library for handling windows, inputs, OpenGL/Vulkan graphics context creation, etc.)

#include <iostream>
#include <fstream>
#include <future>
#include <chrono>
#include <math.h>

#include "imgui.h"

#include "gui.h"
#include "filebrowser.h"
#include "dir.h"
#include "widgets.h"
#include "jpeg.h"
#include "texture.h"
#include "image.h"
#include "photo_process_opts.h"
#include "film.h"
#include "frame.h"
#include "measure.h"
#include "cldriver.h"
#include "color.h"

#include "data.h"

#include <nlohmann/json.hpp>
using json = nlohmann::json;

struct ProcessingOptions
{
    int frame_horz = 0;
    int frame_vert = 0;
    int pixel_x = -1;
    int pixel_y = -1;
};

void show_spectrum(const char * label, const float * spectrum, float scale = 1)
{
    ImGui::Text("%s:", label);
    ImGui::PlotLines("", spectrum, SPECTRUM_SIZE,
        0, nullptr, FLT_MAX, FLT_MAX, ImVec2(0, 80));
    ImGui::SameLine();
    Color xyz = spectrum_to_xyz(spectrum);
    Color srgb = xyz_to_srgb(scale * xyz);
    ImGui::ColorButton(label, *(ImVec4*)&srgb.c, 0, ImVec2(80, 80));
}

class DifpGui : public GuiBuilder
{
public:
    const int SMALL_WIDTH = 1400; // 1280;
    const int SMALL_HEIGHT = 1000; //960;
    enum Mode {
        MODE_IMAGE,
        MODE_GRADIENT
    };

    Rgb32Image processImage(Image& image, const ProcessingOptions& po)
    {
        //std::cout << "(" << image.width << ", " << image.height << ")\n";
        //auto sd = load_spectrum_data("research/profile/wthanson/spectrum.json");
        auto sd = load_spectrum_data(
            std::string("research/profile/wthanson/spectra2/") + m_chosenSpectrum
        );
        //auto pd = load_profile_data("research/profile/wthanson/kodak-vision-250d-5207.json");
        //auto pd = load_profile_data("research/profile/wthanson/kodak-portra-400-portra-400.json");
        //auto pd = load_profile_data("research/profile/wthanson/experim-ill-a.json");
        auto pd = load_profile_data(
            std::string("research/profile/wthanson/profiles/") + m_chosenProfile
        );
        auto opts = UserOptions();
        opts.color_corr = Array<3> {m_red, m_green, m_blue};
        opts.film_exposure = m_filmExposure;
        opts.paper_exposure = m_paperExposure;
        opts.paper_contrast = m_paperContrast;
        opts.curve_smoo = m_curveSmoo;
        //opts.negative = m_filmOnly;
        opts.mode = m_processingMode;
        opts.frame_horz = po.frame_horz;
        opts.frame_vert = po.frame_vert;
        auto processedImage = process_photo(image, sd, pd, opts);
        Rgb32Image img = convert_image_to_rgb32(processedImage);
        return img;
        //tex.load(img);
    }

    void processSmallImage()
    {
        if (!m_imageLoaded) {
            return;
        }
        if (m_inProcessingImage) {
            m_scheduleProcessImage = true;
        } else {
            m_inProcessingImage = true;
            m_processImageFuture = std::async(std::launch::async, [&] {
                ProcessingOptions po;
                po.pixel_x = m_pixelX;
                po.pixel_y = m_pixelY;
                if (m_isCrop) {
                    int sx = m_cropX * m_origImage.width / m_sizeWithFrame.x
                                - SMALL_WIDTH / 2;
                    int sy = m_cropY * m_origImage.height / m_sizeWithFrame.y
                                - SMALL_HEIGHT / 2;
                    if (sx < 0) {
                        sx = 0;
                    }
                    if (sx > m_origImage.width - SMALL_WIDTH - 1) {
                        sx = m_origImage.width - SMALL_WIDTH - 1;
                    }
                    if (sy < 0) {
                        sy = 0;
                    }
                    if (sy > m_origImage.height - SMALL_HEIGHT - 1) {
                        sy = m_origImage.height - SMALL_HEIGHT - 1;
                    }
                    m_smallImage = sub_image(m_origImage, sx, sy, SMALL_WIDTH, SMALL_HEIGHT);
                    po.frame_horz = 0;
                    po.frame_vert = 0;
                    m_sizeWithFrame = ImVec2(SMALL_WIDTH, SMALL_HEIGHT);
                } else {
                    IntSize wh, WH;
                    const auto& format = get_paper_formats()[m_paperFormatIdx];
                    inner_frame(format, float(m_origImage.width) / m_origImage.height,
                                    SMALL_WIDTH, SMALL_HEIGHT, m_frameWidthRatio,
                                    wh, WH);
                    po.frame_horz = WH.width;
                    po.frame_vert = WH.height;
                    m_sizeWithFrame = ImVec2(wh.width + 2 * WH.width,
                                             wh.height + 2 * WH.height);

                    if (wh.width != m_smallImage.width
                     || wh.height != m_smallImage.height)
                    {
                        /*
                        std::cout << "Scaling from "
                                << m_smallImage.width << "x" << m_smallImage.height
                                << " to "
                                << wh.width << "x" << wh.height << "\n";
                        */
                        auto dt = measure([&]{
                            switch (m_mode) {
                            case MODE_IMAGE:
                                m_smallImage = bilinear_scale(m_origImage,
                                                          wh.width, wh.height);
                                break;
                            case MODE_GRADIENT:
                                m_smallImage = gradient(wh.width, wh.height,
                                    Color(0, 0, 0), Color(95.0/300, 100.0/300, 109.0/300));
                                break;
                            }
                        });
                        //std::cout << "Scaling time: " << dt << "s\n";
                    }
                }

                auto image = m_smallImage.clone();
                Rgb32Image img;
                auto dt = measure([&] {
                    img = processImage(image, po);
                });
                //std::cout << "Process time: " << dt << "s\n";
                return img;
            });
        }
    }

    int buildImageWindow()
    {
        ImGui::Begin("Image");
        auto spos = ImGui::GetCursorScreenPos(); 
        int width = m_smallImage.width ? m_smallImage.width : SMALL_WIDTH; 
        int height = m_smallImage.height ? m_smallImage.height : SMALL_HEIGHT;
        ImGui::Image((void*)(intptr_t)tex.id(), m_sizeWithFrame);
        if (ImGui::IsItemClicked()) {
            auto pos = ImGui::GetMousePos();
            int x = pos.x - spos.x;
            int y = pos.y - spos.y;
            //std::cout << "x = " << x << " y = " << y << "\n";
            if (ImGui::GetIO().KeyCtrl) {
                if (!m_isCrop) {
                    m_pixelX = x;
                    m_pixelY = y;
                    auto px = m_smallImage.data[m_smallImage.width * y + x];
                    std::cout << "Clicked on: " << px.to_array() << "\n";
                    processSmallImage();
                }
            } else {
                if (!m_isCrop) {
                    m_cropX = x;
                    m_cropY = y;
                }
                m_isCrop = !m_isCrop;
                processSmallImage();
            }
        }
        auto windowWidth = ImGui::GetWindowSize().x;
        ImGui::End();
        return windowWidth;
    }

    void buildParametersWindow()
    {
        ImGui::Begin("Parameters");
       
        bool endTabBar = ImGui::BeginTabBar("TabBar1"); 
        if (ImGui::BeginTabItem("Params")) {
            if (ImGui::Button("Open File Dialog")) {
                openFileDialog = true;
            }
            ImGui::SameLine();
            if (ImGui::Button("Save")) {
                std::string filename = fi.path();
                filename += m_saveSuffix;
                filename += ".dipp.jpg"; // Exposure and development
                std::cout << "Saving to " << filename << "...\n";
                auto image = m_origImage.clone();
                ProcessingOptions po;
                const PaperFormat& pf = get_paper_formats()[m_paperFormatIdx];
                IntSize sz = outer_frame(pf,
                    image.width, image.height, m_frameWidthRatio);
                po.frame_horz = sz.width;
                po.frame_vert = sz.height;
                auto rgb32Image = processImage(image, po);
                Jpeg jpeg;
                jpeg.save(rgb32Image, filename);
                std::cout << "done\n";
            }
            ImGui::SameLine();
            if (ImGui::Button("Reload")) {
                CLDriver::get().reload();
            }
            ImGui::SameLine();
            if (ImGui::Button("Apply")) {
                processSmallImage();
            }
            if (openFileDialog) {
                if (FileBrowser("Open file", fi)) {
                    if (fi.isConfirmed()) {
                        m_origImage = load_image_from_raw_file(fi.path());
                        m_smallImage.resize(0, 0);
                        m_isCrop = false;
                        m_imageLoaded = true;

                        processSmallImage();

                        selectedPath = fi.path();
                    } else {
                        //selectedPath = "";
                    }
                    openFileDialog = false;
                }
            }
            ImGui::Text("%s", selectedPath.c_str());
            if (ImGui::SliderFloat("Film exposure", &m_filmExposure, -2, 2, "%.2f")) {
                processSmallImage();
            }
            const char* processingModeComboItems[] = {
                "Normal",
                "Negative",
                "Identity",
                "Film exposure",
                "Generated spectrum",
                "Film dev",
                "Paper exposure",
                "Film H < 0",
                "Paper H < 0"
            };
            if (ImGui::IsKeyPressed('O')) {
                m_processingMode = NORMAL;
                processSmallImage();
            } else if (ImGui::IsKeyPressed('N')) {
                m_processingMode = NEGATIVE;
                processSmallImage();
            } else if (ImGui::IsKeyPressed('I')) {
                m_processingMode = IDENTITY;
                processSmallImage();
            } else if (ImGui::IsKeyPressed('F')) {
                m_processingMode = FILM_EXPOSURE;
                processSmallImage();
            } else if (ImGui::IsKeyPressed('G')) {
                m_processingMode = GEN_SPECTR;
                processSmallImage();
            } else if (ImGui::IsKeyPressed('D')) {
                m_processingMode = FILM_DEV;
                processSmallImage();
            } else if (ImGui::IsKeyPressed('P')) {
                m_processingMode = PAPER_EXPOSURE;
                processSmallImage();
            } else if (ImGui::IsKeyPressed('1')) {
                m_processingMode = FILM_NEG_LOG_EXP;
                processSmallImage();
            } else if (ImGui::IsKeyPressed('2')) {
                m_processingMode = PAPER_NEG_LOG_EXP;
                processSmallImage();
            }
            if (ImGui::Combo("Mode", &m_processingMode, processingModeComboItems, 
                    sizeof(processingModeComboItems) / sizeof(const char*))) {
                processSmallImage();
            }
            if (ImGui::SliderFloat("Paper exposure", &m_paperExposure, -5, 5, "%.2f")) {
                processSmallImage();
            }
            /*
            if (ImGui::SliderFloat("Light on paper", &m_lightOnPaper, -5, 5, "%.2f")) {
                processSmallImage();
            }
            */
            if (ImGui::SliderFloat("Red", &m_red, 0, 1.5, "%.3f")) {
                processSmallImage();
            }
            if (ImGui::SliderFloat("Green", &m_green, /*0.3*/0, 1.5 /*0.8*/, "%.3f")) {
                processSmallImage();
            }
            if (ImGui::SliderFloat("Blue", &m_blue, /*0.6*/0, 1.5 /*1.5*/, "%.3f")) {
                processSmallImage();
            }
            /*
            if (ImGui::SliderFloat("Linear AMP", &m_linAmp, -2, 1, "%.2f")) {
                processSmallImage();
            }
            if (ImGui::SliderFloat("P. Red", m_paperFilter, 0, 2, "%.3f")) {
                processSmallImage();
            }
            if (ImGui::SliderFloat("P. Green", m_paperFilter + 1, 0, 2, "%.3f")) {
                processSmallImage();
            }
            if (ImGui::SliderFloat("P. Blue", m_paperFilter + 2, 0, 2, "%.3f")) {
                processSmallImage();
            }
            */
            /*
            if (ImGui::SliderFloat("2nd sub-layer delta", &m_layer2d, 0, 3, "%.2f")) {
                processSmallImage();
            }
            */

            const auto& formats = get_paper_formats();
            std::vector<const char*> items;
            for (const auto& f: formats) {
                items.emplace_back(f.name.c_str());
            }
            if(ImGui::Combo("Paper format", &m_paperFormatIdx,
                items.data(), items.size()))
            {
                //handleFrameChange(formats[m_paperFormatIdx]);
                processSmallImage();
            }
            if (ImGui::SliderFloat("Frame/Width", &m_frameWidthRatio, 0, 0.3, "%.3f")) {
                //handleFrameChange(formats[m_paperFormatIdx]);
                processSmallImage();
            }
            ImGui::InputText("Save suffix", m_saveSuffix, sizeof(m_saveSuffix));
            /*
            if (ImGui::SliderFloat("Film contrast", &m_filmContrast, 0.1, 2, "%.2f")) {
                processSmallImage();
            }
            */
            if (ImGui::SliderFloat("Paper contrast", &m_paperContrast, 0.1, 4, "%.2f")) {
                processSmallImage();
            }
            if (ImGui::SliderFloat("Curve smoothness", &m_curveSmoo, 0.01, 1.0, "%.2f")) {
                processSmallImage();
            }
            /*
            if (ImGui::RadioButton("Image", m_mode == MODE_IMAGE)) {
                m_mode = MODE_IMAGE;
                processSmallImage();
            }
            ImGui::SameLine();
            if (ImGui::RadioButton("Gradient", m_mode == MODE_GRADIENT)) {
                m_mode = MODE_GRADIENT;
                processSmallImage();
            }
            */
            if (MiniBrowser(
                "1",
                [this](){ return m_spectrumDir.files(); },
                m_chosenSpectrum))
            {
                processSmallImage();
            }
            if (MiniBrowser(
                "2",
                [this](){ return m_profileDir.files(); },
                m_chosenProfile))
            {
                processSmallImage();
            }
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Debug")) {
            ImGui::Text("XYZ in: %f, %f, %f",
                m_debug.xyz_in[0],
                m_debug.xyz_in[1],
                m_debug.xyz_in[2]);
            show_spectrum("Spectrum", m_debug.spectrum.data(), 300);
            ImGui::Text("Film exposure: %f, %f, %f",
                m_debug.film_exposure[0],
                m_debug.film_exposure[1],
                m_debug.film_exposure[2]);
            ImGui::Text("Film exposure (log + corr): %f, %f, %f",
                log10(m_debug.film_exposure[0]) + m_filmExposure,
                log10(m_debug.film_exposure[1]) + m_filmExposure,
                log10(m_debug.film_exposure[2]) + m_filmExposure);
            ImGui::Text("Film density: %f, %f, %f",
                m_debug.film_density[0],
                m_debug.film_density[1],
                m_debug.film_density[2]);
            ImGui::Text("Film density2: %f, %f, %f",
                m_debug.film_density2[0],
                m_debug.film_density2[1],
                m_debug.film_density2[2]);
            ImGui::Text("Film true density: %f, %f, %f",
                m_debug.film_tdensity[0],
                m_debug.film_tdensity[1],
                m_debug.film_tdensity[2]);
            show_spectrum("Film fall spectrum", m_debug.film_fall_spectrum.data());
            show_spectrum("Film pass spectrum", m_debug.film_pass_spectrum.data());
            show_spectrum("Film fltr spectrum", m_debug.film_fltr_spectrum.data());
            ImGui::Text("Paper exposure: %f, %f, %f",
                m_debug.paper_exposure[0],
                m_debug.paper_exposure[1],
                m_debug.paper_exposure[2]);
            ImGui::Text("Paper exposure (log + corr): %f, %f, %f",
                log10(m_debug.paper_exposure[0]) + m_paperExposure,
                log10(m_debug.paper_exposure[1]) + m_paperExposure,
                log10(m_debug.paper_exposure[2]) + m_paperExposure);
            ImGui::Text("Paper density: %f, %f, %f",
                m_debug.paper_density[0],
                m_debug.paper_density[1],
                m_debug.paper_density[2]);
            ImGui::Text("Paper true density: %f, %f, %f",
                m_debug.paper_tdensity[0],
                m_debug.paper_tdensity[1],
                m_debug.paper_tdensity[2]);
            show_spectrum("Paper fall spectrum", m_debug.paper_fall_spectrum.data(), 0.2);
            show_spectrum("Paper refl spectrum", m_debug.paper_refl_spectrum.data());
            ImGui::Text("XYZ out: %f, %f, %f",
                m_debug.xyz_out[0],
                m_debug.xyz_out[1],
                m_debug.xyz_out[2]);
            ImGui::Text("sRGB out: %f, %f, %f",
                m_debug.srgb_out[0],
                m_debug.srgb_out[1],
                m_debug.srgb_out[2]);
            ImGui::EndTabItem();
        }
        if (endTabBar) {
            ImGui::EndTabBar();
        }
        ImGui::End();
    }
    
    void build() override
    {
        if (m_inProcessingImage) {
            if (m_processImageFuture.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                m_inProcessingImage = false;

                tex.load(m_processImageFuture.get());
                if (m_scheduleProcessImage) {
                    m_scheduleProcessImage = false;
                    processSmallImage();
                }
            }
        }
        
        /*
        ImGui::SetNextWindowPos(ImVec2(0, 0));
        ImGui::SetNextWindowSize(ImVec2(
            1280 + 2 * ImGui::GetStyle().WindowPadding.x,
            960 + 2 * ImGui::GetStyle().WindowPadding.y
        ));
        */
        ImGui::SetNextWindowPos(ImVec2(0, 0));
        ImGui::SetNextWindowSize(ImVec2(0, 0));
        auto offset = buildImageWindow();

        auto &io = ImGui::GetIO();

        ImGui::SetNextWindowPos(ImVec2(offset, 0));
        ImGui::SetNextWindowSize(ImVec2(io.DisplaySize.x - offset, 0));
        buildParametersWindow();
    }

private:
    FileInfo fi;
    bool openFileDialog = false;
    std::string selectedPath;
    Texture tex;

    bool m_imageLoaded = false;
    Image m_origImage;
    Image m_smallImage;
    float m_filmExposure = 0.0;
    float m_paperExposure = 0.0; //0.1; //-1.48; //-3.18;
    float m_red = 0.0;
    float m_green = 0.0; //0.270; //0.349; //0.57; //0.457; //0.438;
    float m_blue = 0.0; //0.569; //0.614; //0.8; //0.744; //0.765;

    bool m_inProcessingImage = false;
    bool m_scheduleProcessImage = false;
    std::future<Rgb32Image> m_processImageFuture;

    bool m_isCrop = false;
    int m_cropX = 0;
    int m_cropY = 0;
    int m_pixelX = 0;
    int m_pixelY = 0;
    float m_linAmp = 0.0; //428.5;
    bool m_filmOnly = false;
    int m_paperFormatIdx = 10;
    float m_frameWidthRatio = 0;
    ImVec2 m_sizeWithFrame = ImVec2(SMALL_WIDTH, SMALL_HEIGHT);
    char m_saveSuffix[16] = "-pa";
    float m_filmContrast = 1.0;
    float m_paperContrast = 1.0;
    float m_curveSmoo = 0.01; //0.27;
    Debug m_debug;
    Mode m_mode = MODE_IMAGE;
    int m_processingMode = 0;
    float m_paperFilter[3] = {0, 0, 0}; //{0, 0.524, 0.406}; //{0, 0.394, 0.310}; //{0, 0.361, 0.248}; //{0, 0.152, 0.056};
    Dir m_spectrumDir { "research/profile/wthanson/spectra2" };
    std::string m_chosenSpectrum;
    Dir m_profileDir { "research/profile/wthanson/profiles" };
    std::string m_chosenProfile;
};

int main(int, char**)
{
    DifpGui gui;
    return runGui(gui);
}
