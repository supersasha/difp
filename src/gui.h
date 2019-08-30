#pragma once

class GuiBuilder
{
public:
    virtual void build() = 0;
};

int runGui(GuiBuilder& bld);
