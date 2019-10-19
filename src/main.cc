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
#include "jpeg.h"
#include "texture.h"
#include "image.h"
#include "photo_process_opts.h"
#include "film.h"
#include "frame.h"
#include "measure.h"
#include "cldriver.h"
#include "color.h"

#include <json.hpp>
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
        //std::string filmFile = "profiles/film/kodak-portra-400-new-v5.film";
        //std::string paperFile = "profiles/paper/kodak-endura-new-v5.paper";
        std::string filmFile = "profiles/film/kodak-portra-400-experim-v2.film";
        std::string paperFile = "profiles/paper/kodak-endura-experim.paper";

        PhotoProcessOpts opts;
        opts.exposure_correction_film = m_filmExposure;
        opts.exposure_correction_paper = m_paperExposure;

        std::ifstream ffilm(filmFile);
        json jfilm;
        ffilm >> jfilm;
        
        std::ifstream fpaper(paperFile);
        json jpaper;
        fpaper >> jpaper;

        opts.film = jfilm;
        opts.paper = jpaper;
        
        std::ifstream fia("profiles/illuminants/a.illuminant");
        json jia;
        fia >> jia;
        
        std::ifstream fid65("profiles/illuminants/d65.illuminant");
        json jid65;
        fid65 >> jid65;

        opts.illuminant1 = jia;
        opts.illuminant2 = jid65;

        opts.extra.psr = m_red;
        opts.extra.psg = m_green;
        opts.extra.psb = m_blue;
        opts.extra.linear_amp = m_linAmp;
        opts.extra.pixel = po.pixel_y * image.width + po.pixel_x;
        opts.extra.stop = m_filmOnly ? 1 : 0;
        opts.extra.layer2d = m_layer2d;

        opts.extra.frame_horz = po.frame_horz;
        opts.extra.frame_vert = po.frame_vert;
        opts.extra.film_contrast = m_filmContrast;
        opts.extra.paper_contrast = m_paperContrast;
        opts.extra.light_through_film = m_lightThroughFilm;
        opts.extra.light_on_paper = m_lightOnPaper;
        opts.extra.paper_filter[0] = m_paperFilter[0];
        opts.extra.paper_filter[1] = m_paperFilter[1];
        opts.extra.paper_filter[2] = m_paperFilter[2];

        auto processedImage = process_photo(image, opts);
        m_debug = opts.debug;
        Rgb32Image img = convert_image_to_rgb32(processedImage);
        return img;
        //tex.load(img);
    }

    void processSmallImage()
    {
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

                        processSmallImage();

                        selectedPath = fi.path();
                    } else {
                        //selectedPath = "";
                    }
                    openFileDialog = false;
                }
            }
            ImGui::Text("%s", selectedPath.c_str());
            if (ImGui::SliderFloat("Film exposure", &m_filmExposure, -5, 5, "%.2f")) {
                processSmallImage();
            }
            if (ImGui::SliderFloat("Light through film", &m_lightThroughFilm, -5, 5, "%.2f")) {
                processSmallImage();
            }
            if (ImGui::Checkbox("Film only", &m_filmOnly)) {
                processSmallImage();
            }
            if (ImGui::SliderFloat("Paper exposure", &m_paperExposure, -5, 5, "%.2f")) {
                processSmallImage();
            }
            if (ImGui::SliderFloat("Light on paper", &m_lightOnPaper, -5, 5, "%.2f")) {
                processSmallImage();
            }
            if (ImGui::SliderFloat("Red", &m_red, 0, 2, "%.3f")) {
                processSmallImage();
            }
            if (ImGui::SliderFloat("Green", &m_green, /*0.3*/0, 2 /*0.8*/, "%.3f")) {
                processSmallImage();
            }
            if (ImGui::SliderFloat("Blue", &m_blue, /*0.6*/0, 2 /*1.5*/, "%.3f")) {
                processSmallImage();
            }
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
            if (ImGui::SliderFloat("Paper contrast", &m_paperContrast, 0.1, 2, "%.2f")) {
                processSmallImage();
            }
            */
            if (ImGui::RadioButton("Image", m_mode == MODE_IMAGE)) {
                m_mode = MODE_IMAGE;
                processSmallImage();
            }
            ImGui::SameLine();
            if (ImGui::RadioButton("Gradient", m_mode == MODE_GRADIENT)) {
                m_mode = MODE_GRADIENT;
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

    Image m_origImage;
    Image m_smallImage;
    float m_filmExposure = -1.85;
    float m_paperExposure = -3.10; //-1.48; //-3.18;
    float m_red = 0;
    float m_green = 0.270; //0.349; //0.57; //0.457; //0.438;
    float m_blue = 0.569; //0.614; //0.8; //0.744; //0.765;

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
    float m_layer2d = 1.2;
    int m_paperFormatIdx = 10;
    float m_frameWidthRatio = 0;
    ImVec2 m_sizeWithFrame = ImVec2(SMALL_WIDTH, SMALL_HEIGHT);
    char m_saveSuffix[16] = "-15x20-f10";
    float m_filmContrast = 1.0;
    float m_paperContrast = 1.0;
    float m_lightThroughFilm = -1.62;
    float m_lightOnPaper = -1.14;
    Debug m_debug;
    Mode m_mode = MODE_IMAGE;
    float m_paperFilter[3] = {0, 0.372, 0.530}; //{0, 0.524, 0.406}; //{0, 0.394, 0.310}; //{0, 0.361, 0.248}; //{0, 0.152, 0.056};
};

int main(int, char**)
{
    DifpGui gui;
    return runGui(gui);
}
