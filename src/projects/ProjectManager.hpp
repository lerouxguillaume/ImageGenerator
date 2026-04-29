#pragma once
#include <optional>
#include <string>
#include <vector>
#include "Project.hpp"

class ProjectManager {
public:
    explicit ProjectManager(std::string filePath = "projects.json");

    // Create a new project with default settings and persist it.
    Project createProject(const std::string& name);

    // Replace all fields of an existing project (matched by id) and persist.
    // No-op with a log warning if id is not found.
    void updateProject(const Project& project);

    // Remove a project by id and persist. No-op if not found.
    void deleteProject(const std::string& id);

    // Append a new asset type to a project and persist.
    // Returns the new AssetType; logs a warning and returns default if projectId not found.
    AssetType addAssetType(const std::string& projectId,
                           const std::string& name,
                           const Prompt&      promptTokens = {},
                           const AssetConstraints& constraints = {});

    // Replace an existing asset type (matched by id) within a project and persist.
    // No-op with a log warning if either id is not found.
    void updateAssetType(const std::string& projectId, const AssetType& assetType);

    // Remove an asset type from a project and persist. No-op if either id not found.
    void deleteAssetType(const std::string& projectId, const std::string& assetTypeId);

    std::optional<Project>      getProject(const std::string& id) const;
    const std::vector<Project>& getAllProjects() const;

private:
    void     load();
    void     save() const;
    Project* findProject(const std::string& id);

    std::string          filePath_;
    std::vector<Project> projects_;
};
