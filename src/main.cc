// dear imgui: standalone example application for GLFW + OpenGL 3, using programmable pipeline
// If you are new to dear imgui, see examples/README.txt and documentation at the top of imgui.cpp.
// (GLFW is a cross-platform general purpose library for handling windows, inputs, OpenGL/Vulkan graphics context creation, etc.)

#include <iostream>
#include <fstream>
#include <future>
#include <chrono>

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

#include <json.hpp>
using json = nlohmann::json;

struct ProcessingOptions
{
    int frame_horz = 0;
    int frame_vert = 0;
    int pixel_x = -1;
    int pixel_y = -1;
};

class DifpGui : public GuiBuilder
{
public:
    const int SMALL_WIDTH = 1400; // 1280;
    const int SMALL_HEIGHT = 1000; //960;
    Rgb32Image processImage(Image& image, const ProcessingOptions& po) //int x = -1, int y = -1)
    {
        std::string filmFile = "profiles/film/kodak-portra-400-new-v2.film"; //"profiles/film/kodak-portra-400-v5.phm.film";
        std::string paperFile = "profiles/paper/kodak-endura-new-v2.paper"; //"profiles/paper/kodak-endura-experim.phm.paper";

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

        /*
        std::ifstream fextra("profiles/extra.phm");
        json jextra;
        fextra >> jextra;

        opts.extra = jextra;
        */
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

        auto processedImage = process_photo(image, opts);
        Rgb32Image img = convert_image_to_rgb32(processedImage);
        return img;
        //tex.load(img);
    }

    void processSmallImage(int x = -1, int y = -1)
    {
        if (m_inProcessingImage) {
            m_scheduleProcessImage = true;
        } else {
            m_inProcessingImage = true;
            m_processImageFuture = std::async(std::launch::async, [&, x, y] {
                ProcessingOptions po;
                po.pixel_x = x;
                po.pixel_y = y;
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
                            m_smallImage = bilinear_scale(m_origImage,
                                                          wh.width, wh.height);
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
                processSmallImage(x, y);
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
        if (ImGui::SliderFloat("Film exposure", &m_filmExposure, -10, 10, "%.2f")) {
            processSmallImage();
        }
        if (ImGui::SliderFloat("Light through film", &m_lightThroughFilm, -10, 10, "%.2f")) {
            processSmallImage();
        }
        if (ImGui::Checkbox("Film only", &m_filmOnly)) {
            processSmallImage();
        }
        if (ImGui::SliderFloat("Paper exposure", &m_paperExposure, -10, 10, "%.2f")) {
            processSmallImage();
        }
        if (ImGui::SliderFloat("Light on paper", &m_lightOnPaper, -10, 10, "%.2f")) {
            processSmallImage();
        }
        if (ImGui::SliderFloat("Red", &m_red, 0, 10, "%.3f")) {
            processSmallImage();
        }
        if (ImGui::SliderFloat("Green", &m_green, /*0.3*/0, 10 /*0.8*/, "%.3f")) {
            processSmallImage();
        }
        if (ImGui::SliderFloat("Blue", &m_blue, /*0.6*/0, 10 /*1.5*/, "%.3f")) {
            processSmallImage();
        }
        if (ImGui::SliderFloat("Linear AMP", &m_linAmp, 1, 500000000, "%.1f")) {
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
        if (ImGui::SliderFloat("Film contrast", &m_filmContrast, 0.1, 2, "%.2f")) {
            processSmallImage();
        }
        if (ImGui::SliderFloat("Paper contrast", &m_paperContrast, 0.1, 2, "%.2f")) {
            processSmallImage();
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
    float m_filmExposure = -2.5;
    float m_paperExposure = -3.18;
    float m_red = 0.00;
    float m_green = 0.457; //0.438;
    float m_blue = 0.744; //0.765;

    bool m_inProcessingImage = false;
    bool m_scheduleProcessImage = false;
    std::future<Rgb32Image> m_processImageFuture;

    bool m_isCrop = false;
    int m_cropX = 0;
    int m_cropY = 0;
    float m_linAmp = 428.5;
    bool m_filmOnly = false;
    float m_layer2d = 1.2;
    int m_paperFormatIdx = 10;
    float m_frameWidthRatio = 0;
    ImVec2 m_sizeWithFrame = ImVec2(SMALL_WIDTH, SMALL_HEIGHT);
    char m_saveSuffix[16] = "-15x20-f10";
    float m_filmContrast = 1.0;
    float m_paperContrast = 1.0;
    float m_lightThroughFilm = 0.0;
    float m_lightOnPaper = 0.0;
};

int main(int, char**)
{
    DifpGui gui;
    return runGui(gui);
}
