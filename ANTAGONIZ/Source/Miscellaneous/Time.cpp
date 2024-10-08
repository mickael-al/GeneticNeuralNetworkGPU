#include "Time.hpp"
#include "Debug.hpp"
#include <GLFW/glfw3.h>

namespace Ge
{
	Time* Time::s_pInstance = nullptr;

	Time::Time()
	{
		Time::s_pInstance = this;
		m_time = 0.0f;
	}

	void Time::startTime()
	{
		m_startTime = glfwGetTime();
		m_currentTime = m_startTime;
		m_currentTimeF = m_startTime;
	}

	void Time::fixedUpdateTime()
	{
		m_lastTimeF = m_currentTimeF;
		m_currentTimeF = glfwGetTime();
		m_time = static_cast<float>(m_currentTimeF - m_startTime);
		m_fixedDeltaTime = static_cast<float>(m_currentTimeF - m_lastTimeF);
	}

	void Time::updateTime()
	{
		m_lastTime = m_currentTime;
		m_currentTime = glfwGetTime();
		m_deltaTime = static_cast<float>(m_currentTime - m_lastTime);
	}

	void Time::release()
	{
		Time::s_pInstance = nullptr;
	}

	float Time::getDeltaTime() const
	{
		return m_deltaTime;
	}

	float Time::getFixedDeltaTime() const
	{
		return m_fixedDeltaTime;
	}

	float Time::getTime() const
	{
		return m_time;
	}

	float Time::GetFixedDeltaTime()
	{
		return Time::s_pInstance->m_fixedDeltaTime;
	}

	float Time::GetDeltaTime()
	{
		return Time::s_pInstance->m_deltaTime;
	}

	float Time::GetTime()
	{
		return Time::s_pInstance->m_time;
	}
}