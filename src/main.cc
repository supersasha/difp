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

#include <json.hpp>
using json = nlohmann::json;

class DifpGui : public GuiBuilder
{
public:
    const int SMALL_WIDTH = 1280;
    const int SMALL_HEIGHT = 960;
    Rgb32Image processImage(Image& image, int x = -1, int y = -1)
    {
        //std::string filmFile = m_ui.getString("film-file");
        //std::string paperFile = m_ui.getString("paper-file");
        std::string filmFile = "profiles/film/kodak-portra-400-v5.phm.film";
        std::string paperFile = "profiles/paper/kodak-endura-experim.phm.paper";

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

        std::ifstream fextra("profiles/extra.phm");
        json jextra;
        fextra >> jextra;

        opts.extra = jextra;
        opts.extra.psr = m_red;
        opts.extra.psg = m_green;
        opts.extra.psb = m_blue;
        opts.extra.linear_amp = m_linAmp;
        opts.extra.pixel = y * image.width + x;
        std::cout << "x=" << x << " y=" << y << " px: " << opts.extra.pixel << "\n";

        process_photo(image, opts);
        Rgb32Image img = convert_image_to_rgb32(image);
        return img;
        //tex.load(img);
    }

    void processSmallImage(int x = -1, int y = -1)
    {
        if (m_inProcessingImage) {
            m_scheduleProcessImage = true;
        } else {
            m_inProcessingImage = true;
            m_processImageFuture = std::async(std::launch::async, [&, x, y]{
                auto image = m_smallImage.clone();
                return processImage(image, x, y);
            });
        }
    }

    int buildImageWindow()
    {
        ImGui::Begin("Image");
        auto spos = ImGui::GetCursorScreenPos(); 
        ImGui::Image((void*)(intptr_t)tex.id(), ImVec2(m_smallImage.width, m_smallImage.height));//tex.width(), tex.height()));
        if (ImGui::IsItemClicked()) {
            auto pos = ImGui::GetMousePos();
            int x = pos.x - spos.x;
            int y = pos.y - spos.y;
            std::cout << "x = " << x << " y = " << y << "\n";
            if (ImGui::GetIO().KeyCtrl) {
                processSmallImage(x, y);
            } else {
                if (m_isCrop) {
                    m_smallImage = bilinear_scale(m_origImage, SMALL_WIDTH, SMALL_HEIGHT);
                    m_isCrop = !m_isCrop;
                } else {
                    if (m_origImage.width > SMALL_WIDTH
                            && m_origImage.height > SMALL_HEIGHT)
                    {
                        int sx = x * m_origImage.width / m_smallImage.width - SMALL_WIDTH / 2;
                        int sy = y * m_origImage.height / m_smallImage.height - SMALL_HEIGHT / 2;
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
                        std::cout << "showing subimage at: " << sx << ", " << sy << "\n";
                        m_smallImage = sub_image(m_origImage, sx, sy, SMALL_WIDTH, SMALL_HEIGHT);
                        m_isCrop = !m_isCrop;
                    }
                    //std::cout << pos.x - spos.x << ", " << pos.y - spos.y << "\n";
                }

                processSmallImage();
            }
        }
        auto width = ImGui::GetWindowSize().x;
        ImGui::End();
        return width;
    }

    void buildParametersWindow()
    {
        ImGui::Begin("Parameters");

        if (ImGui::Button("Open File Dialog"))
        {
            openFileDialog = true;
        }
        if (openFileDialog) {
            if (FileBrowser("Open file", fi)) {
                if (fi.isConfirmed()) {
                    m_origImage = load_image_from_raw_file(fi.path());
                    m_smallImage = bilinear_scale(m_origImage,
                        SMALL_WIDTH, SMALL_HEIGHT);
                    m_isCrop = false;

                    processSmallImage();

                    selectedPath = fi.path();
                } else {
                    selectedPath = "";
                }
                openFileDialog = false;
            }
        }
        ImGui::Text("%s", selectedPath.c_str());
        if (ImGui::SliderFloat("Film exposure", &m_filmExposure, -5, 5, "%.2f")) {
            processSmallImage();
        }
        if (ImGui::SliderFloat("Paper exposure", &m_paperExposure, -5, 5, "%.2f")) {
            processSmallImage();
        }
        if (ImGui::SliderFloat("Red", &m_red, 0, 3.8, "%.2f")) {
            processSmallImage();
        }
        if (ImGui::SliderFloat("Green", &m_green, 0, 3.8, "%.2f")) {
            processSmallImage();
        }
        if (ImGui::SliderFloat("Blue", &m_blue, 0, 3.8, "%.2f")) {
            processSmallImage();
        }
        if (ImGui::SliderFloat("Linear amplification", &m_linAmp, 1, 500, "%.1f")) {
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
    float m_filmExposure = 0.0;
    float m_paperExposure = -2.0;
    float m_red = 0.00;
    float m_green = 0.31;
    float m_blue = 0.33;

    bool m_inProcessingImage = false;
    bool m_scheduleProcessImage = false;
    std::future<Rgb32Image> m_processImageFuture;

    bool m_isCrop = false;
    float m_linAmp = 166;
};

int main(int, char**)
{
    DifpGui gui;
    return runGui(gui);
}
