#ifndef __FACTORY_H__
#define __FACTORY_H__

#undef min
#undef max
#include "json_struct.h"
#include <map>
#include <functional>
#include <string>
#include <iostream>

template<class B>
class Factory 
{
    std::map<std::string, std::function<B* ()>> s_creators;
    std::vector<std::string> s_names;
public:
    static Factory<B>& getInstance() 
    {
        static Factory<B> s_instance;
        return s_instance;
    }

    template<class T>
    void registerClass(const std::string& name) 
    {
        s_creators.insert({ name, []() -> B* { return new T(); } });
        s_names.push_back(name);
    }

    B* create(const std::string& name) 
    {
        const auto it = s_creators.find(name);
        if (it == s_creators.end()) return nullptr; // not a derived class
        return (it->second)();
    }

    void printRegisteredClasses() 
    {
        for (const auto& creator : s_creators) 
        {
            std::cout << creator.first << '\n';
        }
    }

    const std::map<std::string, std::function<B* ()>>& getDerived() const
    {
        return s_creators;
    }
    const std::vector<std::string> & getDerivedName() const
    {
        return s_names;
    }
};
#define FACTORY(Class) Factory<Class>::getInstance()

template<class B, class T>
class Creator 
{
public:
    explicit Creator(const std::string& name) 
    {
        FACTORY(B).registerClass<T>(name);
    }
};

#define REGISTER(base_class, derived_class, ...) \
  Creator<base_class, derived_class> s_##derived_class##Creator(#derived_class); \
  JS_OBJ_EXT(derived_class, __VA_ARGS__)
  
#endif