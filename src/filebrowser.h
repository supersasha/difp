#pragma once

#include <string>
#include <vector>

struct DirPart
{
    std::string part;
    std::string path;

    DirPart(const std::string& _part, const std::string& _path)
        : part(_part), path(_path)
    {}
};

struct DirEntry
{
    std::string name;
    char type;

    DirEntry(std::string n, char t)
        : name(n), type(t)
    {}
};

class FileInfo
{
public:
    FileInfo();
    
    void recomposeDir(std::vector<DirPart>::const_iterator eit);
    void setDir(const std::string& dir);
    void dirUp();

    const std::vector<DirEntry>& entries() const
    {
        return m_entries;
    }

    std::string filename() const
    {
        return m_filename;
    }

    void setFilename(const std::string& fn)
    {
        m_filename = fn;
    }

    std::string dir() const
    {
        return m_dir;
    }

    void setConfirmed(bool c)
    {
        m_confirmed = c;
    }

    bool isConfirmed() const
    {
        return m_confirmed;
    }

    const std::vector<DirPart>& dirParts() const
    {
        return m_parts;
    }

    const std::string path() const
    {
        return m_dir == "/"
            ? m_dir + m_filename
            : m_dir + "/" + m_filename;
    }
private:

    bool m_confirmed = false;
    
    std::string m_dir;
    std::string m_filename;
    std::vector<DirPart> m_parts;
    std::vector<DirEntry> m_entries;

    void obtainEntries();
};

bool FileBrowser(const std::string& name, FileInfo& fi);
